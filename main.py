#!/usr/bin/env python

"""
    pipenet-pretrained.py
"""

from __future__ import print_function, division

import os
import sys
import json
import argparse
import itertools
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

sys.path.append('/home/bjohnson/projects/simple_ppo')
from models.path import SinglePathPPO
from rollouts import RolloutGenerator

from helpers import set_seeds, to_numpy
from data import make_cifar_dataloaders
from pipenet import PipeNet, AccumulateException
from envs import PretrainedEnv, DummyTrainEnv

from rsub import *
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# --
# Params

def parse_args():
    """ parameters mimic openai/baselines """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--env', type=str, default='pretrained')
    parser.add_argument('--pretrained-path', type=str, default='./models/pipenet-cyclical-20')
    
    parser.add_argument('--total-steps', type=int, default=int(40e6))
    parser.add_argument('--steps-per-batch', type=int, default=64)
    parser.add_argument('--epochs-per-batch', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)
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

dataloaders = make_cifar_dataloaders(train_size=0.9, download=False, seed=np.random.choice(1000))

# --
# Define environment

if args.env == 'pretrained':
    worker = PipeNet(loss_fn=F.cross_entropy).cuda()
    worker.load_state_dict(torch.load(args.pretrained_path))
    env = PretrainedEnv(worker=worker, dataloaders=dataloaders, seed=args.seed)
elif args.env == 'dummy_train':
    worker = PipeNet(loss_fn=F.cross_entropy).cuda()
    env = DummyTrainEnv(worker=worker, dataloaders=dataloaders, seed=args.seed)
else:
    raise Exception('unknown env %s' % args.env, file=sys.stderr)


ppo = SinglePathPPO(
    n_outputs=len(worker.pipes),
    adam_lr=args.adam_lr,
    adam_eps=args.adam_eps,
    entropy_penalty=args.entropy_penalty,
    clip_eps=args.clip_eps,
    cuda=args.cuda,
)
print(ppo, file=sys.stderr)

if args.cuda:
    ppo = ppo.cuda()

roll_gen = RolloutGenerator(
    env=env,
    ppo=ppo,
    steps_per_batch=args.steps_per_batch,
    rms=False,
    advantage_gamma=args.advantage_gamma,
    advantage_lambda=args.advantage_lambda,
    cuda=args.cuda,
    num_workers=args.num_workers, # 8 steps at once
    num_frames=args.num_frames,
    action_dim=len(worker.pipes),
    mode='lever'
)

# --
# Run

start_time = time()
ppo_epoch = 0
history = []
while roll_gen.step_index < args.total_steps:
    
    # --
    # Sample a batch of rollouts
    
    roll_gen.next()
    
    mean_reward = roll_gen.batch['rewards'].mean()
    action_counts = list(roll_gen.batch['actions'].cpu().numpy().mean(axis=0))
    
    history.append({
        "ppo_epoch"     : ppo_epoch,
        "step_index"    : roll_gen.step_index,
        "mean_reward"   : mean_reward,
        "action_counts" : action_counts,
        "elapsed_time"  : time() - start_time,
        
        "train_total_batches" : env.dataloaders['train'].total_batches,
        "val_total_batches"   : env.dataloaders['val'].total_batches,
        "test_total_batches"  : env.dataloaders['test'].total_batches,
        
        "train_epochs" : env.dataloaders['train'].epochs,
        "val_epochs"   : env.dataloaders['val'].epochs,
        "test_epochs"  : env.dataloaders['test'].epochs,
        
    })
    
    print(history[-1])
    
    # --
    # Update model parameters
    
    ppo.backup()
    for epoch in range(args.epochs_per_batch):
        minibatches = roll_gen.iterate_batch(
            batch_size=args.batch_size * args.num_workers,
            seed=(epoch, roll_gen.step_index)
        )
        for minibatch in minibatches:
            _ = ppo.step(**minibatch)
    
    ppo_epoch += 1

# --
# Inspect

_ = plt.plot([h['mean_reward'] for h in history][:300])
show_plot()

action_counts = np.vstack([h['action_counts'] for h in history])[:500]
for i, r in enumerate(action_counts.T):
    _ = plt.plot(r, alpha=0.5, label=i)

plt.legend(loc='lower right')
show_plot()


