import os
import argparse
from model import load_text_generation_model
from data_load import dataset_loader
from utils import make_folders
import deepspeed
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from deepspeed.utils import safe_get_full_fp32_param, safe_get_full_grad,safe_set_full_fp32_param
from deepspeed import comm as dist
import torch
import evaluate as eevaluate
import time
from peft import PeftModel
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup, AutoModelForCausalLM 

from torch.nn.parallel import DistributedDataParallel as DDP

from utils import generate_response
import numpy as np
from tensor_selector import selection_DP, downscale_t_dy_and_t_dw
from tensor_flops import compute_tensor_flops, compute_forward_flops
from utils import flops_counter, compute_squad_metric

#++++++++++++++++++++++++++++++++++++++++++++++++
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
#++++++++++++++++++++++++++++++++++++++++++++++++

#green trainer training function
def train_green(
        args,is_mixed_precision="no",seed:int =42
    ):
        #in order to make sure that the initialization is same
        set_seed(seed)
        

        if is_mixed_precision=="no":#don't use mixed precision training
            accelerator = Accelerator(log_with="tensorboard", project_dir=args.log_dir)
        else:
            mixed_precision="fp16"#use mixed precision training
            accelerator = Accelerator(mixed_precision=mixed_precision,log_with="tensorboard", project_dir=args.log_dir)

        #track during training
        config = {
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "train_type": args.train_type,
        }
        accelerator.init_trackers("baselines", config=config)

        #load un_wrapped model
        model = load_text_generation_model(
        args.model_name, args.train_type,
        output_attentions=False,
    )
        print("[info] un_wrapped model have been loaded************************")
   
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
        model,optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model,optimizer, train_loader, val_loader, lr_scheduler
    )
        print("all the things prepared!")
        
        #set up how many epochs to use tensor selection once
        interval=2

        if accelerator.is_main_process:#is main rank
            t_dy, t_dw = compute_tensor_flops(
                model=model,
                model_name=args.model_name,
                input_length=args.max_input_length,
                #output_length=args.max_output_length,
                batch_size=args.batch_size,
                draw_figure=False,
            )
            #print(f"t_dy is:{t_dy}********************************")
            #print(f"t_dw is:{t_dw}********************************")

            t_fp = compute_forward_flops(
                model=model,
                model_name=args.model_name,
                input_length=args.max_input_length,
                #output_length=args.max_output_length,
                batch_size=args.batch_size,
            )
            #accelerator.print(f"t_fp is:{t_fp}********************************")

            #dowmscale
            t_dy_q, t_dw_q, disco = downscale_t_dy_and_t_dw(t_dy, t_dw, Tq=1e3)

            #reverse for backward
            t_dy_q = np.flip(t_dy_q)
            t_dw_q = np.flip(t_dw_q)

            #process the constraint you set up for computing
            def to_backward_rho(rho, t_fp, t_dy, t_dw):
                t_bp = np.sum(t_dy + t_dw)
                rho_bp = rho * (1 + t_fp / t_bp) - t_fp / t_bp
                if rho_bp <= 0:
                    rho_bp = 0.05
                    rho_reset = (rho_bp + t_fp / t_bp) / (1 + t_fp / t_bp)
                    print(f"rho is too low. rho has been reset to {rho_reset}")
                return rho_bp
        

            rho_bp = to_backward_rho(args.rho, t_fp, t_dy, t_dw)
            N = t_dw.shape[0]
            T = np.sum(t_dw + t_dy) # maximally possible BP time
            T_limit = rho_bp * T
            t_dy_cumsum = 0
            t_dy_flipped = np.flip(t_dy)
            for k in range(N):
                t_dy_cumsum += t_dy_flipped[k]
                if t_dy_cumsum > T_limit:
                    break
            N_limit = N - k
            accelerator.print(f"N: {N}, N_limit: {N_limit}")
            N_limit=torch.tensor(N_limit).cuda(device=0)
            N=torch.tensor(N).cuda(device=0)
            
            
        else:
            for param_idx, (name, param) in enumerate(model.named_parameters()):
                param= safe_get_full_fp32_param(param)
            for param_idx, (name, param) in enumerate(model.named_parameters()):
                param= safe_get_full_fp32_param(param)
            N_limit=torch.tensor(0).cuda(device=dist.get_rank())
            N=torch.tensor(0).cuda(device=dist.get_rank())

        dist.broadcast(N_limit,src=0)
        dist.broadcast(N,src=0)

        #dist.barrier()
        accelerator.wait_for_everyone()

        total_time = 0
        
        for epoch in range(args.num_epochs):
            t_start = time.time()
            
            model.train()
            total_loss = 0

            if epoch % interval == 0:#do tensor selection
                for idx, (_, param) in enumerate(model.named_parameters()):
                    param.requires_grad = True
                print("#### Selecting trainable tensors...")
                data_iter = iter(train_loader)
                batch = next(data_iter)
                
                if accelerator.is_main_process:
                    # cache original weight values if it is main rank
                    w_0=[]
                    for _, param in model.named_parameters():
                        param=safe_get_full_fp32_param(param)
                        w_0.append(param.clone().detach().cpu())

                else:
                    for _, param in model.named_parameters():
                        param=safe_get_full_fp32_param(param)

                accelerator.wait_for_everyone()
                #Now we hava w0
                
                for idx, (_, param) in enumerate(model.named_parameters()):
                        if idx < N_limit.item():
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                
                accelerator.wait_for_everyone()
                outputs = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                        )
                loss = outputs.loss
                accelerator.backward(loss)

                # perform update
                optimizer.step()
                
                # lr_scheduler.step()
                optimizer.zero_grad()
                
                for idx, (_, param) in enumerate(model.named_parameters()):
                    param.requires_grad = True

                if accelerator.is_main_process:
                    # cache updated weight values
                    w_1=[]
                    for _, param in model.named_parameters():
                        param=safe_get_full_fp32_param(param)
                        #accelerator.print(f"{param}")
                        w_1.append(param.clone().detach().cpu())

                    # compute weight changes, it takes optimizer's schedule into account
                    dw_0 = [w_1_k - w_0_k for (w_0_k, w_1_k) in zip(w_0, w_1)]

                    del w_1
                    del w_0

                else:
                    for _, param in model.named_parameters():
                        param=safe_get_full_fp32_param(param)
               
                accelerator.wait_for_everyone()
                #Now we have dw0

                for idx, (_, param) in enumerate(model.named_parameters()):
                        if idx < N_limit.item():
                            param.requires_grad = False
                        else:
                            param.requires_grad = True

                accelerator.wait_for_everyone()

                # cache gradients
                outputs = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                        )

                loss = outputs.loss
                accelerator.backward(loss)
                
                accelerator.wait_for_everyone()

                if accelerator.is_main_process:
                    grad_1=[]
                    for _, param in model.named_parameters():
                        param=safe_get_full_grad(param)
                        if param is not None:
                            grad_1.append(param.clone().detach().cpu())
                        else:
                            grad_1.append(torch.tensor(0.0).cpu())
                else:
                    for _, param in model.named_parameters():
                        param=safe_get_full_grad(param)

                accelerator.wait_for_everyone()

                optimizer.step()

                # lr_scheduler.step()
                optimizer.zero_grad()
                
                #del model
                #model = load_text_generation_model(
                #args.model_name, args.train_type,
                #output_attentions=False,
                #)
                #model = load_state_dict_from_zero_checkpoint(model, "ck/saved_models/save_model.pth")
                #model=accelerator.prepare(model)

                accelerator.wait_for_everyone()

                if accelerator.is_main_process:
                    I = [torch.sum((grad_1_k * dw_0_k)) for (grad_1_k, dw_0_k) in zip(grad_1, dw_0)]
                    del grad_1
                    del dw_0
                    I = torch.tensor(I)
                    I = I / torch.max(torch.abs(I))

                #accelerator.print(f"I is:{I}********************************")

                    I = -I.numpy()
                    I = np.flip(I)
                    # print("disco:", disco)

                    max_importance, m = selection_DP(t_dy_q, t_dw_q, I, rho=rho_bp)
                    m = np.flip(m)
                    print(f"m:{m}, length of m : {len(m)}")
                    print("max importance:", max_importance)
                    print("%T_sel:", 100 * np.sum(np.maximum.accumulate(m) * t_dy + m * t_dw) / np.sum(t_dy + t_dw))

                    mm=torch.arange(N.item())
                    for i in range(N.item()):
                        mm[i]=torch.tensor(m[i])
                    mm=mm.cuda(device=0)
                else:
                    mm=torch.arange(N.item())
                    mm=mm.cuda(device=dist.get_rank())
                
                accelerator.wait_for_everyone()
                dist.broadcast(mm,src=0)
                accelerator.wait_for_everyone()

                # ground trainability
                for k, (_, param) in enumerate(model.named_parameters()):
                    if mm[k].item()== 1:
                            param.requires_grad = True
                    else:
                            param.requires_grad = False
                del mm

            
            for step, batch in enumerate(tqdm(train_loader)):
                accelerator.wait_for_everyone()

                #*****no need to load input and target to GPU(accelarator will do this)
                outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )

                loss=outputs.loss#this loss is one batch_size/GPUs loss
                #print(f"loss is {loss}")
                loss_whole=accelerator.gather(loss).mean()#gather all the loss on all GPUs

                #print loss one batch
                accelerator.print(f" epoch {epoch},train loss is {loss_whole} ********************************")

                #compute the total loss of train_loader
                total_loss += loss_whole.detach().float()

                accelerator.wait_for_everyone()

                #replace loss.backward
                accelerator.backward(loss)

                optimizer.step()

                lr_scheduler.step()

                optimizer.zero_grad()

            t_end = time.time()
            epoch_time = t_end - t_start
            print(f"Epoch Time: {epoch_time} (s)")
            total_time += epoch_time
            print(f"Total Time: {total_time} (s)")
        
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
            
            accelerator.wait_for_everyone()

        #save model
        model.save_checkpoint("~/saved_models/save_model.pth")
        accelerator.print("model has been saved successfully!********************************")
        
        print(f"Total GreenTrainer Time: {total_time} (s)")     
        accelerator.end_training()

#normal training_function using accelerate (not green trainer)
def train(
        args,is_mixed_precision="no",seed:int =42
    ):  
        #in order to make sure that the initialization is same
        set_seed(seed)

        if is_mixed_precision=="no":#don't use mixed precision training
            accelerator = Accelerator(log_with="tensorboard", project_dir=args.log_dir)
        else:
            mixed_precision="fp16"#use mixed precision training
            accelerator = Accelerator(mixed_precision=mixed_precision,log_with="tensorboard", project_dir=args.log_dir)
        
        #track during training
        config = {
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "train_type": args.train_type,
        }
        accelerator.init_trackers("baselines", config=config)

        #load un_wrapped model
        model = load_text_generation_model(
        args.model_name, args.train_type,
        output_attentions=False,
    )
        print("[info] un_wrapped model have been loaded************************")
   
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

        #load test dataloader
        test_loader, _ = dataset_loader(
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
        model,optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model,optimizer, train_loader, val_loader, lr_scheduler
    )
        print("all the things prepared!")

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

                loss_whole=accelerator.gather(loss).sum()#gather all the loss on all GPUs

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
                loss=accelerator.gather(loss).sum()

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

        
            #_runtime_evaluate(model,val_loader,args,accelerator,tokenizer)

            #save model
            model.save_checkpoint("~/saved_models/save_model.pth")
            accelerator.print("model has been saved successfully!********************************")
        

        print(f"Total Time: {total_time} (s)")
        print("training finished****************")

        #_runtime_evaluate(model,test_loader,args,accelerator,tokenizer)

        accelerator.end_training()

#evaluate function during training time
def _runtime_evaluate(model,dataset,args,accelerator,tokenizer):
        model.eval()
        # for summarization
        m_rouge1 = 0
        m_rouge2 = 0
        m_rougeL = 0
        m_rougeLsum = 0
        # for question answering
        m_f1 = 0
        m_em = 0
        rouge_metric = eevaluate.load('rouge')
        total_count = 0
        with torch.no_grad():
            for step, batch in enumerate(tqdm(dataset)):
                batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                batch_size = batch['input_ids'].shape[0]
                    
                all_results = generate_response(
                    model, 
                    args.train_type,
                    tokenizer, 
                    batch['lp_sources'], batch['labels'], batch['input_ids_lens'],
                    max_length=args.max_output_length
                )
                
                summarization_results = rouge_metric.compute(predictions=all_results["outputs_text"], references=all_results["labels_text"])
                qa_results = compute_squad_metric(tokenizer, predictions=all_results["outputs_tokens"], references=all_results["labels_tokens"])
                
                m_rouge1 += (summarization_results['rouge1'] * batch_size)
                m_rouge2 += (summarization_results['rouge2'] * batch_size)
                m_rougeL += (summarization_results['rougeL'] * batch_size)
                m_rougeLsum += (summarization_results['rougeLsum'] * batch_size)
                m_f1 += (qa_results["f1"] * batch_size)
                m_em += (qa_results["EM"] * batch_size)
                
                total_count += batch_size
        
        m_rouge1 /= total_count
        m_rouge2/= total_count
        m_rougeL /= total_count
        m_rougeLsum /= total_count
        m_f1 /= total_count
        m_em /= total_count
        if accelerator.is_local_main_process:
            print(f"On validation/test set, rouge1={100*m_rouge1}, rouge2={100*m_rouge2}, rougeL={100*m_rougeL}, rougeLsum={100*m_rougeLsum}")
            print(f"On validation/test set, f1={m_f1}%, EM={m_em}%")

#final evaluation
def evaluate(test_loader,args,tokenizer):
        #model=_load_model_fsdp(args)

        #model=_load_model(args)
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

#save model when use fsdp
def _save_model_fsdp(model,accelerator,args):
    unwarpped_model=accelerator.unwrap_model(model)
    full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(unwarpped_model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
         state = accelerator.get_state_dict(unwarpped_model)
    unwarpped_model.save_pretrained(
              args.model_path,
              is_main_process=accelerator.is_main_process,
              save_function=accelerator.save,
              state_dict=state
      )

#load model when use fsdp   
def _load_model_fsdp(args):
        #if args.train_type in ["lora", "adalora", "prefix_tuning"]:
            #model = PeftModel.from_pretrained(args.model, args.model_path)
            #if args.train_type == "lora" or args.train_type == "adalora":
                 #args.model.merge_and_unload()
        #else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path)
        
        return model

#save model when use DeepSpeed
def _save_model_ds(model,accelerator,args):
    success=model.save_checkpoint(args.model_path)
    if success:
       print(f"Success ********************************")
    else:
       print(f"Failure ********************************")

#load model when use fsdp   
def _load_model_ds(model,accelerator,args):
    unwrapped_model = accelerator.unwrap_model(model)
    fp32_model = load_state_dict_from_zero_checkpoint(unwrapped_model, args.model_path)
    print("success ********************************")
     
def _save_model(model,args):
    model.save_pretrained(args.model_path)
    
def _load_model(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    return model


if __name__=='__main__':
   parser = argparse.ArgumentParser(description='parser for training decoder-only models')
   parser.add_argument('--model_name', type=str, default='facebook/opt-1.3b', help='opt and bloomz series')
   parser.add_argument('--tokenizer', type=str, default='facebook/opt-1.3b', help='tokenizer to use')
   parser.add_argument('--dataset_name', type=str, default='scitldr', help='scitldr or dialogsum')
   parser.add_argument('--scheme', type=str, default='green trainer', help='baselines or green_trainer')
   parser.add_argument('--train_type', type=str, default='green trainer', help='green trainer')
   parser.add_argument('--max_input_length', type=int, default=512, help='number of input tokens for causal language modeling')
   parser.add_argument('--max_output_length', type=int, default=128, help='number of new output tokens for generation')
   parser.add_argument('--batch_size', type=int, default=4, help='batch size during training and generation')
   parser.add_argument('--learning_rate', type=float, default=2e-5, help='learning_rate during training')
   parser.add_argument('--num_epochs', type=int, default=5, help='num_epochs during training')
   parser.add_argument('--log_dir', type=str, default=f"logs/{'facebook/opt-1.3b'.replace('/', '_')}_{'full_finetuning'}", help='log-dir during training and generation')
   parser.add_argument('--task', type=str, default='train', help='train or evaluate')
   parser.add_argument('--model_path', type=str, default= f"saved_models/{'facebook/opt-1.3b'.replace('/', '_')}_{'full_finetuning'}", help='model_path to save during training and generation')
   parser.add_argument('--rho', type=int, default=0.4, help='speedup ratio for GreenTrainer')

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   parser.add_argument('--nodes',default=1,type=int,help='the number of nodes/computer')
   parser.add_argument('--gpus', type=int, default=4, help='the number of gpus per computer to use')
   parser.add_argument('--world_size', type=int, default=4, help='the number of all the gpus to use')
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

   args = parser.parse_args()
   print("[info] args has been parsed**********************")

   make_folders("logs", "saved_models")
   print("[info] folders to save and write logs have been created************************")

   os.environ["TOKENIZERS_PARALLELISM"] = "true"

   if args.task == 'train':#to train
        #train(args,"yes")
        train_green(args)

   else:#to evaluate
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

