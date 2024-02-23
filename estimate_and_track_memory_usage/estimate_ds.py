from model import load_text_generation_model
from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live

#This file is used to estimate the gpu memory usage when you use the DeepSpeed library

#with actual model
model=load_text_generation_model('facebook/opt-6.7b','full_finetuning')
estimate_zero2_model_states_mem_needs_all_live(model, num_gpus_per_node=4, num_nodes=1)

#without actual model but total parameters
from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_cold
estimate_zero2_model_states_mem_needs_all_cold(total_params=67e8, num_gpus_per_node=4, num_nodes=1)

#zero stage 3
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=4, num_nodes=1)

#stage3 without actual model
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_cold
estimate_zero3_model_states_mem_needs_all_cold(total_params=2851e6, largest_layer_params=32e6, num_gpus_per_node=8, num_nodes=1)
