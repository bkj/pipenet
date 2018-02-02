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
from torch.nn import functional as F
from torch.autograd import Variable

from base_model import BaseModel
from data import make_mnist_dataloaders
from helpers import set_seeds, Flatten

# torch.set_default_tensor_type('torch.DoubleTensor')

# --
# Model

class StandardNet(BaseModel):
    def __init__(self, lr=1e-3, loss_fn=F.cross_entropy):
        super(StandardNet, self).__init__(loss_fn=loss_fn)
        
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
# Losses

def onehot(y, n_categories):
    y = y.view(-1, 1)
    oh = torch.Tensor(y.shape[0], n_categories).cuda()
    oh.zero_()
    oh.scatter_(1, y.data, 1)
    return Variable(oh)


def l1_prob_loss(x, y):
    return ((x - y).abs()).sum(dim=-1)


def lp_prob_loss(x, y, p=2.0):
    return ((x - y) ** p).sum(dim=-1)


def linf_prob_loss(x, y):
    return (x - y).abs().max(dim=-1)[0]


# !! This assumes you run `F.log_softmax` beforehand
def crossentropy_loss(x, y, p=1.0, avg=False):
    return -((y * x).sum(dim=-1) ** p)


def categorical_hinge_loss(x, y, margin=1.0, p=1.0):
    pos = (x * y).sum(dim=-1)
    neg = ((1 - y) * x).max(dim=-1)[0]
    return ((neg - pos + margin).clamp(min=0) ** p)


def categorical_hinge_loss_2(x, y, margin=1.0, p=1.0, avg=False):
    pos = (x * y).sum(dim=-1)
    neg = ((1 - y) * (1 - x)).clamp(min=0).sum(dim=-1)
    return (pos + neg) ** p


def loss_wrapper(post_fn, loss_fn):
    def f(x, y):
        loss = loss_fn(post_fn(x), onehot(y, n_categories=10))
        if np.isnan(loss.data.sum()):
            print(x)
            print(y)
            print(loss)
            raise Exception
        
        return loss.mean()
    
    return f

# post-processing functions
eye_fn = lambda x: x
softmax = lambda x: F.softmax(x, dim=-1)
log_softmax = lambda x: F.log_softmax(x, dim=-1)
sigmoid = lambda x: F.sigmoid(x, dim=-1)


# --
# Run

standard_results = []

loss_fn = loss_wrapper(softmax, crossentropy_loss)

worker = StandardNet(loss_fn=loss_fn, lr=1e-3).cuda()
for epoch in range(100):
    train_score = worker.step(dataloaders)
    standard_results.append({
        "epoch"        : epoch,
        "train_score"  : train_score,
        "val_score"    : worker.evaluate(dataloaders, mode='val'),
        "test_score"   : worker.evaluate(dataloaders, mode='test'),
    })
    print(standard_results[-1])


