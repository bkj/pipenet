#!/usr/bin/env python

"""
    ablatenet.py
"""

from __future__ import print_function, division

import sys
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
from torch import nn
from torch.autograd import Variable

from base_model import BaseModel
from data import make_mnist_dataloaders
from helpers import set_seeds, Flatten, ablate

# --
# Model

class AblateStandardNet(BaseModel):
    def __init__(self, lr=1e-3):
        super(AblateStandardNet, self).__init__()
        
        self.layers = nn.Sequential(*[
            nn.Conv2d(1, 32, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            Flatten(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 10),
        ])
        
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.score = {}
        
    def forward(self, x):
        return self.layers(x)
    
    def ablate_forward(self, x, p=0.5):
        x = self.layers[0](x)
        x = self.layers[1](x)
        x = self.layers[2](x)
        x = self.layers[3](x)
        x = ablate(x, p=p)
        x = self.layers[4](x)
        x = self.layers[5](x)
        x = self.layers[6](x)
        x = self.layers[7](x)
        x = self.layers[8](x)
        x = ablate(x, p=p)
        x = self.layers[9](x)
        x = self.layers[10](x)
        x = self.layers[11](x)
        return x
    
    def evaluate(self, dataloaders, mode='val'):
        loader = dataloaders[mode]
        if loader is None:
            return None, None
        else:
            _ = self.eval()
            correct, ablate_correct, total = 0, 0, 0
            for data, target in loader:
                data = Variable(data.cuda(), volatile=True)
                
                output = self(data)
                pred = output.data.max(1, keepdim=True)[1].cpu()
                correct += pred.eq(target.view_as(pred)).sum()
                
                ablate_output = self.ablate_forward(data)
                ablate_pred = ablate_output.data.max(1, keepdim=True)[1].cpu()
                ablate_correct += ablate_pred.eq(target.view_as(pred)).sum()
                
                total += data.shape[0]
                
            self.score[mode] = correct / total
            self.score['ablate_' + mode] = ablate_correct / total
            return self.score[mode], self.score['ablate_' + mode]

# --
# IO

set_seeds()

dataloaders = make_mnist_dataloaders(train_size=1.0)

# --
# Run

standard_results = []

worker = AblateStandardNet(lr=1e-3).cuda()
for epoch in range(100):
    _ = worker.step(dataloaders)
    train_score, ablate_train_score = worker.evaluate(dataloaders, mode='train')
    val_score, ablate_val_score     = worker.evaluate(dataloaders, mode='val')
    test_score, ablate_test_score   = worker.evaluate(dataloaders, mode='test')
    
    standard_results.append({
        "epoch"               : epoch,
        
        "train_score"         : train_score,
        "val_score"           : val_score,
        "test_score"          : test_score,
        
        "ablate_train_score"  : ablate_train_score,
        "ablate_val_score"    : ablate_val_score,
        "ablate_test_score"   : ablate_test_score,
    })
    print(standard_results[-1])



pd.DataFrame(standard_results).to_csv('./standard_results.tsv', sep='\t', index=False)

