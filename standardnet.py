#!/usr/bin/env python

"""
    standardnet.py
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
from helpers import set_seeds, Flatten

# --
# Model

class StandardNet(BaseModel):
    def __init__(self, lr=1e-3):
        super(StandardNet, self).__init__()
        
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

# --
# IO

set_seeds(123)

dataloaders = make_mnist_dataloaders(train_size=0.9)

# --
# Run

standard_results = []

worker = StandardNet(lr=1e-3).cuda()
for epoch in range(100):
    train_score = worker.step(dataloaders)
    standard_results.append({
        "epoch"        : epoch,
        "train_score"  : train_score,
        "val_score"    : worker.evaluate(dataloaders, mode='val'),
        "test_score"   : worker.evaluate(dataloaders, mode='test'),
    })
    print(standard_results[-1])


