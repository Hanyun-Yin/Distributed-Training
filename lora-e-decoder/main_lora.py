import os
import gc
import psutil
import threading
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

#++++++++++++++++++++++++++++++++++++++++++++++++
from accelerate import Accelerator
from accelerate.utils import set_seed
#++++++++++++++++++++++++++++++++++++++++++++++++

from peft import LoraConfig, TaskType, get_peft_model
from peft import PeftModel,PeftConfig


def levenshtein_distance(str1, str2):
    # TC: O(N^2)
    # SC: O(N^2)
    if str1 == str2:
        return 0
    num_rows = len(str1) + 1
    num_cols = len(str2) + 1
    dp_matrix = np.empty((num_rows, num_cols))
    dp_matrix[0, :] = range(num_cols)
    dp_matrix[:, 0] = range(num_rows)

    for i in range(1, num_rows):
        for j in range(1, num_cols):
            if str1[i - 1] == str2[j - 1]:
                dp_matrix[i, j] = dp_matrix[i - 1, j - 1]
            else:
                dp_matrix[i, j] = min(dp_matrix[i - 1, j - 1], dp_matrix[i - 1, j], dp_matrix[i, j - 1]) + 1

    return dp_matrix[num_rows - 1, num_cols - 1]

# Converting Bytes to Megabytes
def b2mb(x):
    return int(x / 2**20)

#evaluation using one gpu(not distributed)
def evaluate(test_loader,args,tokenizer):
        model = load_text_generation_model(
        args.model_name, args.train_type,
        output_attentions=False,
    )
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

# This context manager is used to track the peak memory usage of the process
class TorchTracemalloc:
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.begin = torch.cuda.memory_allocated()
        self.process = psutil.Process()

        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)
        # print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")

#distributed lora training function
def main1():
    peft_config = LoraConfig(
            peft_type="LORA", 
            task_type="SEQ_2_SEQ_LM", 
            inference_mode=False, 
            r=8, lora_alpha=32, target_modules=["q", "v"],
            lora_dropout=0.1,
        )
    parser = argparse.ArgumentParser(description='parser for training decoder-only models')
    parser.add_argument('--model_name', type=str, default="google/flan-t5-xxl", help='opt and bloomz series')
    parser.add_argument('--tokenizer', type=str, default="google/flan-t5-xxl", help='tokenizer to use')
    parser.add_argument('--dataset_name', type=str, default='scitldr', help='scitldr or dialogsum')
    parser.add_argument('--scheme', type=str, default='baselines', help='baselines or green_trainer')
    parser.add_argument('--train_type', type=str, default='lora', help='full_finetuning or lora')
    parser.add_argument('--max_input_length', type=int, default=512, help='number of input tokens for causal language modeling')
    parser.add_argument('--max_output_length', type=int, default=64, help='number of new output tokens for generation')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size during training and generation')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='learning_rate during training')
    parser.add_argument('--num_epochs', type=int, default=5, help='num_epochs during training')
    parser.add_argument('--log_dir', type=str, default=f"logs/{'facebook/opt-6.7b'.replace('/', '_')}_{'full_finetuning'}", help='log-dir during training and generation')
    parser.add_argument('--task', type=str, default='train', help='train or evaluate')
    parser.add_argument('--model_path', type=str, default= f"saved_models/{'facebook/opt-6.7b'.replace('/', '_')}_{'full_finetuning'}", help='model_path to save during training and generation')
    parser.add_argument('--prefix', type=str, default='summarize: ', help='prefix of input strings')
    args = parser.parse_args()

    seed = 42
    set_seed(seed)

    
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
    #model_name=args.model_name,
    max_input_length=args.max_input_length, 
    max_output_length=args.max_output_length,
    batch_size=args.batch_size,
    prefix=args.prefix,
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
    #model_name=args.model_name,
    max_input_length=args.max_input_length, 
    max_output_length=args.max_output_length,
    batch_size=args.batch_size,
    prefix=args.prefix,
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

#distributed lora evaluating function
def main2():
    accelerator=Accelerator()
    #peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"],lora_dropout=0.1)
    parser = argparse.ArgumentParser(description='parser for training decoder-only models')
    parser.add_argument('--model_name', type=str, default="google/flan-t5-xxl", help='opt and bloomz series')
    parser.add_argument('--tokenizer', type=str, default="google/flan-t5-xxl", help='tokenizer to use')
    parser.add_argument('--dataset_name', type=str, default='scitldr', help='scitldr or dialogsum')
    parser.add_argument('--scheme', type=str, default='baselines', help='baselines or green_trainer')
    parser.add_argument('--train_type', type=str, default='lora', help='full_finetuning or lora')
    parser.add_argument('--max_input_length', type=int, default=512, help='number of input tokens for causal language modeling')
    parser.add_argument('--max_output_length', type=int, default=64, help='number of new output tokens for generation')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size during training and generation')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='learning_rate during training')
    parser.add_argument('--num_epochs', type=int, default=5, help='num_epochs during training')
    parser.add_argument('--log_dir', type=str, default=f"logs/{'facebook/opt-6.7b'.replace('/', '_')}_{'full_finetuning'}", help='log-dir during training and generation')
    parser.add_argument('--task', type=str, default='train', help='train or evaluate')
    parser.add_argument('--prefix', type=str, default='summarize: ', help='prefix of input strings')
    parser.add_argument('--model_path', type=str, default= f"saved_models/{'facebook/opt-6.7b'.replace('/', '_')}_{'full_finetuning'}", help='model_path to save during training and generation')
    args = parser.parse_args()
    test_loader, tokenizer = dataset_loader(
    dataset_name=args.dataset_name,
    split="test",
    tokenizer_name=args.model_name,
    #model_name=args.model_name,
    max_input_length=args.max_input_length, 
    max_output_length=args.max_output_length,
    batch_size=args.batch_size,
    prefix=args.prefix,
    shuffle=True,
    keep_in_memory=False,
    print_info=False,
)

    print("[info] test_loader have been created************************")

    model = load_text_generation_model(
        args.model_name, args.train_type,
        output_attentions=False,
    )

    config = PeftConfig.from_pretrained("~/saved_models/save_model.pth")
    model=PeftModel.from_pretrained(model, "~/saved_models/save_model.pth")

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
                batch['input_ids'], batch['labels'],
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
    main1()