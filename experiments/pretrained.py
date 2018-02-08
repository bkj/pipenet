#!/usr/bin/env python

"""
    path-main.py
"""

from __future__ import print_function

import os
import sys
import json
import argparse
import itertools
import numpy as np
import pandas as pd
from time import time
from itertools import cycle
from collections import OrderedDict

import torch
from torch import nn
from torch.autograd import Variable

import gym
from gym.spaces.box import Box

sys.path.append('/home/bjohnson/projects/simple_ppo')
sys.path.append('/home/bjohnson/projects/simple_ppo/external')
from models import SinglePathPPO
from rollouts import RolloutGenerator
from monitor import Monitor
from subproc_vec_env import SubprocVecEnv

from helpers import set_seeds, to_numpy
from data import make_mnist_dataloaders
from standardnet import StandardNet

# torch.set_default_tensor_type('torch.DoubleTensor') # Necessary?
# torch.set_default_tensor_type('torch.FloatTensor')

from rsub import *
from matplotlib import pyplot as plt

# --
# Params

def parse_args():
    """ parameters mimic openai/baselines """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--total-steps', type=int, default=int(40e6))
    parser.add_argument('--steps-per-batch', type=int, default=200)
    parser.add_argument('--epochs-per-batch', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--num-frames', type=int, default=4)
    
    parser.add_argument('--advantage-gamma', type=float, default=0.99)
    parser.add_argument('--advantage-lambda', type=float, default=0.95)
    
    parser.add_argument('--clip-eps', type=float, default=0.2)
    parser.add_argument('--adam-eps', type=float, default=1e-5)
    parser.add_argument('--adam-lr', type=float, default=1e-3)
    parser.add_argument('--entropy-penalty', type=float, default=0.001)
    
    parser.add_argument('--seed', type=int, default=987)
    
    parser.add_argument('--cuda', action="store_true")
    parser.add_argument('--log-dir', type=str, default="./logs")
    
    return parser.parse_args()

# --
# Args

args = parse_args()

set_seeds(args.seed)

# --
# IO

dataloaders = make_mnist_dataloaders(train_size=0.9, mode='fashion_mnist', 
    download=True, pretensor=True, seed=np.random.choice(1000))

# --
# Run

standard_results = []

worker = StandardNet(lr=1e-3, dropout=[0.5, 0.5], kernel_size=[3, 3]).cuda()
# worker2 = StandardNet(lr=1e-3, dropout=[0.5, 0.5], kernel_size=[3, 3]).cuda()

_ = worker.train()
start_time = time()
for epoch in range(100):
    train_score = worker.step(dataloaders, num_batches=211)
    standard_results.append({
        "epoch"        : epoch,
        "train_score"  : train_score,
        "val_score"    : worker.evaluate(dataloaders, mode='val'),
        "test_score"   : worker.evaluate(dataloaders, mode='test'),
        "elapsed_time" : time() - start_time,
    })
    print(standard_results[-1])

standard_results = pd.DataFrame(standard_results)
standard_results.max(axis=0)

standard_results.to_csv('./_results/standard_results-55-33.tsv', sep='\t', index=False)

# epoch          99.000000
# test_score      0.933600
# train_score     0.966426
# val_score       0.941500
# dtype: float64
