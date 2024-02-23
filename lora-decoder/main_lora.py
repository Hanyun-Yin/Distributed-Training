import os
import argparse
from model import load_text_generation_model
from data_load import dataset_loader

import deepspeed
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from deepspeed import comm as dist
import torch
import evaluate as eevaluate
import time
from tqdm import tqdm

from transformers import get_linear_schedule_with_warmup, AutoModelForCausalLM
from transformers import AutoConfig
from utils import flops_counter, compute_squad_metric

from utils import generate_response
import numpy as np

from accelerate import Accelerator
from accelerate.utils import set_seed

from peft import LoraConfig, TaskType, get_peft_model
from peft import PeftModel,PeftConfig

#evaluation using one gpu(not distributed)
def evaluate(test_loader,args,tokenizer):
        
        model = load_text_generation_model(
        args.model_name, args.train_type,
        output_attentions=False,
    )
        #load model from checkpoint
        model = load_state_dict_from_zero_checkpoint(model, "~/saved_models/save_model.pth")
        model = model.cuda()
        
        model.eval()
        # for summarization
        m_rouge1 = 0
        m_rouge2 = 0
        m_rougeL = 0
        m_rougeLsum = 0
        # for question answering
        m_f1 = 0
        m_em = 0
        
        total_count = 0
        with torch.no_grad():
            for step, batch in enumerate(tqdm(test_loader)):
                batch = {k: v.cuda() for k, v in batch.items()}
                batch_size = batch['input_ids'].shape[0]
                    
                all_results = generate_response(
                    model, 
                    args.train_type,
                    tokenizer, 
                    batch['lp_sources'], batch['labels'], batch['input_ids_lens'],
                    max_length=args.max_output_length
                )
                rouge_metric=eevaluate.load("rouge")
                summarization_results = rouge_metric.compute(predictions=all_results["outputs_text"], references=all_results["labels_text"])
                qa_results = compute_squad_metric(tokenizer, predictions=all_results["outputs_tokens"], references=all_results["labels_tokens"])
                
                m_rouge1 += (summarization_results['rouge1'] * batch_size)
                m_rouge2 += (summarization_results['rouge2'] * batch_size)
                m_rougeL += (summarization_results['rougeL'] * batch_size)
                m_rougeLsum += (summarization_results['rougeLsum'] * batch_size)
                m_f1 += (qa_results["f1"] * batch_size)
                m_em += (qa_results["EM"] * batch_size)
                print(f"On test set, f1={m_f1}%, EM={m_em}%,rouge1={m_rouge1}, rouge2={m_rouge2}, rougeL={m_rougeL},rougeLsum={m_rougeLsum}********************************")
                
                total_count += batch_size
        
        m_rouge1 /= total_count
        m_rouge2/= total_count
        m_rougeL /= total_count
        m_rougeLsum /= total_count
        m_f1 /= total_count
        m_em /= total_count
        print(f"On test set, rouge1={100*m_rouge1}, rouge2={100*m_rouge2}, rougeL={100*m_rougeL}, rougeLsum={100*m_rougeLsum}")
        print(f"On test set, f1={m_f1}%, EM={m_em}%")

#distributed lora training function
def train(
    args,is_mixed_precision="no",seed:int =42
):
    accelerator = Accelerator(log_with="tensorboard", project_dir=args.log_dir)

    #track during training
    config = {
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "train_type": args.train_type,
    }
    accelerator.init_trackers("baselines", config=config)

    #load train_dataloader and tokenizer(tokenizers below are the same)
    train_loader, tokenizer = dataset_loader(
    dataset_name=args.dataset_name,
    split="train",
    tokenizer_name=args.model_name,
    model_name=args.model_name,
    max_input_length=args.max_input_length, 
    batch_size=args.batch_size,
    shuffle=True,
    keep_in_memory=False,
    print_info=False,
)
    print("[info] train_loader have been created************************")

    #load validation dataloader
    val_loader, _ = dataset_loader(
    dataset_name=args.dataset_name,
    split="validation",
    tokenizer_name=args.model_name,
    model_name=args.model_name,
    max_input_length=args.max_input_length, 
    batch_size=args.batch_size,
    shuffle=False,
    keep_in_memory=False,
    print_info=False,
)
    print("[info] val_loader have been created************************")

    # creating model
    model = load_text_generation_model(
        args.model_name, args.train_type,
        output_attentions=False,
    )
    #use lora
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"],lora_dropout=0.1)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    #use accelerator to wrap the model
    #*****warning：better to wrap the model before the initiation of optimizer
    #accelerator.prepare(model)

    #initialize the optimizer and lr_scheduler
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_loader) * args.num_epochs),
    )

    #wrap the optimizer, train_loader, val_loader, lr_scheduler
    model,optimizer, train_loader, val_loader,lr_scheduler = accelerator.prepare(
    model,optimizer, train_loader, val_loader,lr_scheduler
)
    accelerator.print("all the things prepared***")

    is_ds_zero_3 = False
    if getattr(accelerator.state, "deepspeed_plugin", None):
        is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3

    total_time = 0
    for epoch in range(args.num_epochs):
        t_start = time.time()
        
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_loader)):
            #*****no need to load input and target to GPU(accelarator will do this)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss=outputs.loss#this loss is one batch_size/GPUs loss

            loss_whole=accelerator.gather(loss).mean()#gather all the loss on all GPUs

            #print loss one batch
            accelerator.print(f" epoch {epoch},train loss is {loss_whole} ********************************")

            #compute the total loss of train_loader
            total_loss += loss_whole.detach().float()

            #replace loss.backward
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        #print time of one epoch
        t_end = time.time()
        epoch_time = t_end - t_start
        accelerator.print(f"Epoch Time: {epoch_time} (s)")
        total_time += epoch_time
        accelerator.print(f"Total Time: {total_time} (s)")
        
        #do evaluation during training time
        model.eval()
        eval_loss = 0
        for step, batch in enumerate(tqdm(val_loader)):
            #*****no need to load input and target to GPU(accelarator will do this)
            with torch.no_grad():
                outputs = model(
                    input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                )

            #gather all the outputs to compute the loss
            loss = outputs.loss
            loss=accelerator.gather(loss).mean()

            eval_loss += loss.detach().float()

        #compute the average train and evaluate loss
        eval_epoch_loss = eval_loss / len(val_loader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(train_loader)
        train_ppl = torch.exp(train_epoch_loss)
        
        if accelerator.is_local_main_process:#is main rank
            #write the logs
            metrics = {
                'Loss/train': train_epoch_loss.item(),
                'PPL/train': train_ppl.item(),
                'Loss/valid': eval_epoch_loss.item(),
                'PPL/valid': eval_ppl.item(),
            }
            accelerator.log(metrics, step=epoch)
            accelerator.print(f"epoch={epoch} train_ppl={train_ppl.item()} train_loss={train_epoch_loss.item()} eval_ppl={eval_ppl.item()} eval_loss={eval_epoch_loss.item()}")
        
        #synchronize
        accelerator.wait_for_everyone()

        #save model
        #model.save_checkpoint("~/saved_models/save_model.pth")
        #accelerator.print("model has been saved successfully!********************************")

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)

    unwrapped_model.save_pretrained(
        "~/saved_models/save_model.pth",
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
    )

    print(f"Total Time: {total_time} (s)")
    print("training finished****************")

    accelerator.end_training()

def main():
    parser = argparse.ArgumentParser(description='parser for training decoder-only models')
    parser.add_argument('--model_name', type=str, default='facebook/opt-13b', help='opt and bloomz series')
    parser.add_argument('--tokenizer', type=str, default='facebook/opt-13b', help='tokenizer to use')
    parser.add_argument('--dataset_name', type=str, default='scitldr', help='scitldr or dialogsum')
    parser.add_argument('--scheme', type=str, default='baselines', help='baselines or green_trainer')
    parser.add_argument('--train_type', type=str, default='lora', help='full_finetuning or lora')
    parser.add_argument('--max_input_length', type=int, default=512, help='number of input tokens for causal language modeling')
    parser.add_argument('--max_output_length', type=int, default=64, help='number of new output tokens for generation')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size during training and generation')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='learning_rate during training')
    parser.add_argument('--num_epochs', type=int, default=5, help='num_epochs during training')
    parser.add_argument('--task', type=str, default='train', help='train, evaluate or distributed evaluate')
    args = parser.parse_args()

    #=====================================================================================================
    parser.add_argument('--nodes',default=1,type=int,help='the number of nodes/computer')
    parser.add_argument('--gpus', type=int, default=4, help='the number of gpus per computer to use')
    parser.add_argument('--world_size', type=int, default=4, help='the number of all the gpus to use')
    #=====================================================================================================
    if args.task == 'train':#to train
       train(args,"no")
    else:#to not distributedly evaluate 
        test_loader, tokenizer= dataset_loader(
        dataset_name=args.dataset_name,
        split="test",
        tokenizer_name=args.model_name,
        model_name=args.model_name,
        max_input_length=args.max_input_length, 
        batch_size=args.batch_size,
        shuffle=False,
        keep_in_memory=False,
        print_info=False,
    )
        print("[info] test_loader have been created************************")
        evaluate(test_loader,args,tokenizer)

#distributed lora evaluating function
def main2():
    accelerator=Accelerator()
    parser = argparse.ArgumentParser(description='parser for training decoder-only models')
    parser.add_argument('--model_name', type=str, default='facebook/opt-13b', help='opt and bloomz series')
    parser.add_argument('--tokenizer', type=str, default='facebook/opt-13b', help='tokenizer to use')
    parser.add_argument('--dataset_name', type=str, default='scitldr', help='scitldr or dialogsum')
    parser.add_argument('--scheme', type=str, default='baselines', help='baselines or green_trainer')
    parser.add_argument('--train_type', type=str, default='lora', help='full_finetuning or lora')
    parser.add_argument('--max_input_length', type=int, default=512, help='number of input tokens for causal language modeling')
    parser.add_argument('--max_output_length', type=int, default=64, help='number of new output tokens for generation')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size during training and generation')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='learning_rate during training')
    parser.add_argument('--num_epochs', type=int, default=5, help='num_epochs during training')
    parser.add_argument('--task', type=str, default='train', help='train or evaluate')

    args = parser.parse_args()
    test_loader, tokenizer= dataset_loader(
        dataset_name=args.dataset_name,
        split="test",
        tokenizer_name=args.model_name,
        model_name=args.model_name,
        max_input_length=args.max_input_length, 
        batch_size=args.batch_size,
        shuffle=False,
        keep_in_memory=False,
        print_info=False,
    )

    print("[info] test_loader have been created************************")
    model = load_text_generation_model(
        args.model_name, args.train_type,
        output_attentions=False,
    )

    config = PeftConfig.from_pretrained("PATH")
    model=PeftModel.from_pretrained(model, "PATH")

    #wrap the optimizer, train_loader, val_loader, lr_scheduler
    model,test_loader= accelerator.prepare(
    model,test_loader
)

    model.eval()
    # for summarization
    m_rouge1 = 0
    m_rouge2 = 0
    m_rougeL = 0
    m_rougeLsum = 0
    # for question answering
    m_f1 = 0
    m_em = 0
    
    total_count = 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_loader)):
            #batch = {k: v.cuda() for k, v in batch.items()}
            batch_size = batch['input_ids'].shape[0]
            #batch_size=args.batch_size
                
            all_results = generate_response(
                model, 
                args.train_type,
                tokenizer, 
                batch['lp_sources'], batch['labels'], batch['input_ids_lens'],
                max_length=args.max_output_length
            )
            rouge_metric=eevaluate.load("rouge")
            summarization_results = rouge_metric.compute(predictions=all_results["outputs_text"], references=all_results["labels_text"])
            qa_results = compute_squad_metric(tokenizer, predictions=all_results["outputs_tokens"], references=all_results["labels_tokens"])
            
            m_rouge1 += (accelerator.gather(torch.tensor(summarization_results['rouge1']).cuda(device=dist.get_rank())).mean() * batch_size)
            m_rouge2 += (accelerator.gather(torch.tensor(summarization_results['rouge2']).cuda(device=dist.get_rank())).mean() * batch_size)
            m_rougeL += (accelerator.gather(torch.tensor(summarization_results['rougeL']).cuda(device=dist.get_rank())).mean() * batch_size)
            m_rougeLsum += (accelerator.gather(torch.tensor(summarization_results['rougeLsum']).cuda(device=dist.get_rank())).mean() * batch_size)
            m_f1 += (qa_results["f1"] * batch_size)
            m_em += (qa_results["EM"] * batch_size)
            accelerator.print(f"On test set, f1={m_f1}%, EM={m_em}%,rouge1={m_rouge1}, rouge2={m_rouge2}, rougeL={m_rougeL},rougeLsum={m_rougeLsum}********************************")
            
            total_count += batch_size
    
    m_rouge1 /= total_count
    m_rouge2/= total_count
    m_rougeL /= total_count
    m_rougeLsum /= total_count
    m_f1 /= total_count
    m_em /= total_count
    print(f"On test set, rouge1={100*m_rouge1}, rouge2={100*m_rouge2}, rougeL={100*m_rougeL}, rougeLsum={100*m_rougeLsum}")
    print(f"On test set, f1={m_f1}%, EM={m_em}%")
 


if __name__ == "__main__":
    main()