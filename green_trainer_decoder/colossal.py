
from model import load_text_generation_model
import torch
from torch.optim import SGD
from torchvision.models import resnet18
import torch.nn as nn
import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import TorchDDPPlugin,GeminiPlugin
from colossalai.zero import GeminiOptimizer
from colossalai.nn.optimizer import HybridAdam
from colossalai.zero import ColoInitContext
from transformers import T5Tokenizer, T5ForConditionalGeneration

def train():
    colossalai.launch_from_torch(
    config={}
)
    plugin = GeminiPlugin()
    booster = Booster(plugin=plugin)
    #model = load_text_generation_model(
        #'google/flan-t5-base', 'full_finetuning',
        #output_attentions=False,
    #)
    #model=resnet18()
    #tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", device_map="auto")
    criterion = lambda x: x.mean()
    optimizer = HybridAdam((model.parameters()), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    model, optimizer, criterion, _, scheduler = booster.boost(model, optimizer, criterion, lr_scheduler=scheduler)
    #print(model)
    #for param_idx, (name, param) in enumerate(model.named_parameters()):
        #param=nn.Parameter(torch.tensor(0.0))
    for param_idx, (name, param) in enumerate(model.named_parameters()):   
        print(f"param_idx is :{param_idx},name is :{name},param is: {param}*****")
        
    
train()


