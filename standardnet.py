#!/usr/bin/env python

"""
    standardnet.py
"""

from __future__ import print_function, division

import sys
import numpy as np
import pandas as pd
from time import time
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
    def __init__(self, lr=1e-3, dropout=[0.5, 0.5], kernel_size=[3, 5]):
        super(StandardNet, self).__init__()
        
        self.layers = nn.Sequential(*[
            nn.Conv2d(1, 32, kernel_size=kernel_size[0], padding=int((kernel_size[0] - 1) / 2)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=kernel_size[1], padding=int((kernel_size[1] - 1) / 2)),
            nn.Dropout2d(p=dropout[0]),
            nn.MaxPool2d(2),
            nn.ReLU(),
            Flatten(),
            nn.Linear(3136, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout[1]),
            nn.Linear(128, 10),
        ])
        
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        
        self.score = {}
        
    def forward(self, x):
        return self.layers(x)

if __name__ == '__main__':
    # --
    # IO

    set_seeds(123)

    dataloaders = make_mnist_dataloaders(train_size=0.9, pretensor=True)

    # --
    # Run

    standard_results = []

    worker = StandardNet(lr=1e-3).cuda()
    for epoch in range(100):
        t = time()
        train_score = worker.step(dataloaders)
        standard_results.append({
            "epoch"        : epoch,
            "train_score"  : train_score,
            "val_score"    : worker.evaluate(dataloaders, mode='val'),
            "test_score"   : worker.evaluate(dataloaders, mode='test'),
            "time"         : time() - t,
        })
        print(standard_results[-1])


