from collections import OrderedDict

import torch
from torch import nn as nn
from transformers import AutoTokenizer, AutoModel

class WeekChallenge(nn.Module):
    
    def __init__(self, model_args):
        super(WeekChallenge, self).__init__()

        self.args = model_args

        self.tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")
        self.embed = AutoModel.from_pretrained("monologg/kobigbird-bert-base")        
        
        if self.args["model_size"] == "base":
            self.regressor = BaseRegressor()
        elif self.args["model_size"] == "small":
            self.regressor = SmallRegressor()
        elif self.args["model_size"] == "large":
            self.regressor = LargeRegressor()

        if self.args["use_cuda"]:
            self.regressor = self.regressor.cuda()
            self.embed = self.embed.cuda()

    def forward(self, x):
        
        with torch.no_grad():
            token_sequence = self.tokenizer(x,return_tensors="pt", padding=True)

            if self.args["use_cuda"]:
                token_sequence.to("cuda")
            
            repre = self.embed(**token_sequence)
        
        x1 = repre[0][:,0:1,:].squeeze(1).detach()
        x1 = self.regressor(x1)
        
        return x1
    
class SmallRegressor(nn.Module):
    
    def __init__(self):
        super(SmallRegressor, self).__init__()
        self.regressor = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(768, 256)),
            ('siu1', nn.SiLU()),
            ('linear2', nn.Linear(256, 5)),
        ]))

    def forward(self, x):
        x = self.regressor(x)
        return x

class BaseRegressor(nn.Module):
    
    def __init__(self):
        super(BaseRegressor, self).__init__()
        self.regressor = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(768, 256)),
            ('silu1', nn.SiLU()),
            ('linear2', nn.Linear(256, 128)),
            ('tanh', nn.Tanh()),
            ('linear3', nn.Linear(128, 5))
        ]))
        
    def forward(self, x):
        x = self.regressor(x)
        return x

class LargeRegressor(nn.Module):
    
    def __init__(self):
        super(LargeRegressor, self).__init__()
        self.regressor = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(768, 512)),
            ('silu1', nn.SiLU()),
            ('linear2', nn.Linear(512, 256)),
            ('silu2', nn.SiLU()),
            ('linear3', nn.Linear(256, 128)),
            ('tanh', nn.Tanh()),
            ('linear4', nn.Linear(128, 5))
        ]))
        
    def forward(self, x):
        x = self.regressor(x)
        return x