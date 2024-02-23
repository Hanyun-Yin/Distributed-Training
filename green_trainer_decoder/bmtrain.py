import bmtrain as bmt
import time
bmt.init_distributed(seed=0)

import torch
import torch.nn as nn
from model_center.model import opt, BertConfig
from model_center.layer import Linear

class BertModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = Bert.from_pretrained("bert-base-uncased")
        self.dense = Linear(config.dim_model, 2)
        bmt.init_parameters(self.dense)

    def forward(self, input_ids, attention_mask):
        pooler_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        logits = self.dense(pooler_output)
        return logits

config = BertConfig.from_pretrained("bert-base-uncased")
model = BertModel(config)
#bmt.print_rank(model)
bmt.init_parameters(model)
#bmt.print_rank(model)
#for param_idx, (name, param) in enumerate(model.named_parameters()): 
    #param.data=nn.Parameter(torch.tensor(0.0))
    #param[:]=0
for param_idx, (name, param) in enumerate(model.named_parameters()):
    bmt.print_rank(f"param_idx is :{param_idx},name is :{name},param is: {param}*****")
if bmt.rank()==0:
    time.sleep(10)
    for param_idx, (name, param) in enumerate(model.named_parameters()):
        bmt.print_rank(f"param_idx is :{param_idx},name is :{name},param is: {param}*****")
bmt.synchronize()
print(f"rank {bmt.rank()} is OK********************************")
    
