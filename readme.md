# **Multi-GPU code for Full-finetuning, LoRA, and Green Trainer.**

## Introduction

This is the code repository for distributed Full finetuning, LoRA and Green Trainer. All the code is based on Accelerate and DeepSpeed library.

## Requirements

To use this code repository, you need to install all the libraries in the requirements.txt file.

## General Usage

### To use full-finetuning:

1.change the args in the main_FT.py of the full-finetuning folder.

2.write the specific dataloader and evaluate function according to the task

3.change the configure file in the config_file folder to choose the strategy you want to use.You can directly use configure file of accelerate library(see the config_file folder of my code, choose to use the file named DS.yaml) :

                `zero_stage`: [0] Disabled, [1] optimizer state partitioning, [2]                 optimizer+gradient state partitioning and [3] optimizer+gradient+parameter                 partitioning
                `gradient_accumulation_steps`: Number of training steps to accumulate                 gradients before averaging and applying them.
                `gradient_clipping`: Enable gradient clipping with value.
               `offload_optimizer_device`: [none] Disable optimizer offloading, [cpu]                 offload optimizer to CPU, [nvme] offload optimizer to NVMe SSD. Only                 applicable with ZeRO >= Stage-2.
                `offload_param_device`: [none] Disable parameter offloading, [cpu] offload                 parameters to CPU, [nvme] offload parameters to NVMe SSD. Only applicable                 with ZeRO Stage-3.
                `zero3_init_flag`: Decides whether to enable `deepspeed.zero.Init` for                 constructing massive models. Only applicable with ZeRO Stage-3.
                `zero3_save_16bit_model`: Decides whether to save 16-bit model weights                 when using ZeRO Stage-3.
                `mixed_precision`: `no` for FP32 training, `fp16` for FP16 mixed-precision                 training and `bf16` for BF16 mixed-precision training.

            Further, to use some strategies that accelerate library doesn't support,use DS_config_with_path.yaml together with DS_config.json. To change, you can rewrite the DS_config.json. To see how to change, you can turn to this web page:[DeepSpeed Configuration JSON - DeepSpeed](https://www.deepspeed.ai/docs/config-json/)

4.command input: accelerate launch –-path_of_config_file  path_of_main.py

### To use LoRA:

        Use lora-decoder folder just as full-finetuning if the model you want to use is based on decoder architecture.The lora-e-decoder folder is for model based on encoder-decoder architecture.

### To use Green Trainer:

    change the "rho" of the args.(rho means the speedup ratio. For example, if you set that rho=0.4, it means that the trainable parameters of the model reduce to 0.4*original model parameters) 

    Other settings to use this code are the same as fine-tuning.

## Citation

    @article{huang2023towards,
    title={Towards Green AI in Fine-tuning Large Language Models via Adaptive Backpropagation},
    author={Huang, Kai and Yin, Hanyun and Huang, Heng and Gao, Wei},
    journal={arXiv preprint arXiv:2309.13192},
    year={2023}

}
