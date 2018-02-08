#!/usr/bin/env python

"""
    path-main.py
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
from itertools import cycle
from collections import OrderedDict, Counter

import torch
from torch import nn
from torch.autograd import Variable

import gym
from gym.spaces.box import Box

sys.path.append('/home/bjohnson/projects/simple_ppo')
sys.path.append('/home/bjohnson/projects/simple_ppo/external')
from models.path import SinglePathPPO
from rollouts import RolloutGenerator
from monitor import Monitor
from subproc_vec_env import SubprocVecEnv

from helpers import set_seeds, to_numpy
from data import make_cifar_dataloaders
from pipenet import PipeNet, AccumulateException

from rsub import *
from matplotlib import pyplot as plt

# --
# Params

def parse_args():
    """ parameters mimic openai/baselines """
    parser = argparse.ArgumentParser()
    
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
# Environment

class PipeNetEnv(object):
    def __init__(self, worker, dataloaders, seed=123):
        self.worker = worker
        self._pipes = np.array(list(worker.pipes.keys()))
        
        self.dataloaders = {
            "train" : self._iterwrapper(dataloaders['train'], mode='train'),
            "val"   : self._iterwrapper(dataloaders['val'], mode='val'),
            "test"  : self._iterwrapper(dataloaders['test'], mode='test'),
        }
        
        self.rs = np.random.RandomState(seed=seed)
        self.train = True
        self.epochs = Counter()
        self.steps = 0
    
    def reset(self):
        return self._random_state()
    
    def _random_state(self):
        return self.rs.normal(0, 1, (8, 32))
    
    def _iterwrapper(self, loader, mode):
        data = list(iter(loader))
        X = torch.cat([t[0] for t in data])
        y = torch.cat([t[1] for t in data])
        
        while True:
            chunks = torch.chunk(torch.randperm(X.shape[0]), X.shape[0] // loader.batch_size)
            for chunk in chunks:
                yield X[chunk], y[chunk]
            
            self.epochs[mode] += 1
    
    def _test(self, mask, mode='val'):
        
        _ = self.worker.eval()
        
        data, target = next(self.dataloaders[mode])
        data = Variable(data, volatile=True).cuda()
        
        mask_pipes = [tuple(pipe) for pipe in self._pipes[mask == 1]]
        self.worker.set_pipes(mask_pipes)
        
        try:
            output = self.worker(data)
            return (to_numpy(output).argmax(axis=1) == to_numpy(target)).mean()
        except AccumulateException:
            return 0.0
        except:
            raise
    
    def step(self, actions):
        payout = []
        for action in actions:
            payout.append(self._test(action, mode='test'))
        
        payout = np.array(payout).reshape(-1, 1)
        
        is_done = np.array([False] * actions.shape[0]).reshape(-1, 1)
        state = self._random_state()
        
        self.steps += 1
        
        return state, payout, is_done, None


# --
# Args

args = parse_args()

set_seeds(args.seed)

# --
# IO

dataloaders = make_cifar_dataloaders(train_size=0.9, download=False, seed=np.random.choice(1000))

# --
# Define environment

worker = PipeNet().cuda()
worker.load_state_dict(torch.load('./models/pipenet-cyclical-20'))

worker.graph[(64, 256)][0].conv1.weight.data *= 10
worker.graph[(64, 512)][0].conv1.weight.data *= 10
worker.graph[(128, 512)][0].conv1.weight.data *= 10

env = PipeNetEnv(worker=worker, dataloaders=dataloaders, seed=123)

# >>

actions = np.random.choice((0, 1), (8, len(worker.pipes)))
_ = env.step(actions)

z = [str(bin(i)).split('b')[-1] for i in range(2 ** len(worker.pipes))]
z = ['0' * (6 - len(zz)) + zz for zz in z]
z = np.vstack(map(list, z)).astype(int)

payout_list = []
for _ in tqdm(range(64)):
    _, payout, _, _ = env.step(z)
    payout_list.append(payout.squeeze())

payouts = np.vstack(payout_list).mean(axis=0)

zs = z[np.argsort(-payouts)]
ps = payouts[np.argsort(-payouts)]

zs[:10]
ps[:10]

# !! Upshot is the spread between models w/ and w/o connections is _very_ small
# !! What is the norm of the real paths vs. the untrained branches?

# <<

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
    action_counts = roll_gen.batch['actions'].cpu().numpy().mean(axis=0)
    
    history.append({
        "ppo_epoch" : ppo_epoch,
        "step_index" : roll_gen.step_index,
        "mean_reward" : mean_reward,
        "action_counts" : action_counts,
        "elapsed_time" : time() - start_time,
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

