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
from collections import OrderedDict, Counter, deque

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

from rsub import *
from matplotlib import pyplot as plt

# --
# Params

# !! Evaluate test accuracy after each epoch
# !! Try w/ fully convolutional networks instead of linear layers

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--total-steps', type=int, default=int(40e6))
    parser.add_argument('--steps-per-batch', type=int, default=200) # steps_per_batch * num_workers = number of batches between PPO updates
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
    
    parser.add_argument('--seed', type=int, default=123)
    
    parser.add_argument('--cuda', action="store_true")
    parser.add_argument('--log-dir', type=str, default="./logs")
    
    return parser.parse_args()

# --
# Initialize

class PathEnv(object):
    def __init__(self, workers, dataloaders, seed=123):
        self.workers = workers
        self.dataloaders = {
            "train" : self._iterwrapper(dataloaders['train'], mode='train'),
            "val"   : self._iterwrapper(dataloaders['val'], mode='val'),
            "test"  : self._iterwrapper(dataloaders['test'], mode='test'),
        }
        
        self.rs = np.random.RandomState(seed=seed)
        self.train = True
        self.epochs = Counter()
        self.acc_buffer = deque(maxlen=512)
    
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
    
    def _joint_forward(self, data, mask):
        for i, m in enumerate(mask):
            data = self.workers[m].layers[i](data)
        
        return data
    
    def _step(self, mask):
        
        for worker in self.workers.values():
            _ = worker.train()
            _ = worker.opt.zero_grad()
        
        data, target = next(self.dataloaders['train'])
        data, target = Variable(data.cuda()), Variable(target.cuda())
        
        output = self._joint_forward(data, mask)
        loss = self.workers[0].loss_fn(output, target)
        loss.backward()
        
        for worker in self.workers.values():
            _ = worker.opt.step()
        
        return self._test(mask, mode='val')
    
    def _test(self, mask, mode='val'):
        
        for worker in self.workers.values():
            _ = worker.eval()
        
        data, target = next(self.dataloaders[mode])
        data = Variable(data, volatile=True).cuda()
        
        output = self._joint_forward(data, mask)
        
        # Should _definitely_ be detrending this, especially at the beginning
        acc = (to_numpy(output).argmax(axis=1) == to_numpy(target)).mean()
        
        self.acc_buffer.append(acc)
        acc_mean = np.mean(self.acc_buffer)
        acc_sd = np.std(self.acc_buffer)
        return (acc - acc_mean) / (acc_sd + 1e-5)
        
    def step(self, actions):
        payout = []
        for action in actions:
            if self.train:
                payout.append(self._step(action))
            else:
                payout.append(self._test(action, mode='test'))
        
        payout = np.array(payout).reshape(-1, 1)
        
        is_done = np.array([False] * actions.shape[0]).reshape(-1, 1)
        state = self._random_state()
        return state, payout, is_done, None

# --
# Args

args = parse_args()
set_seeds(args.seed)

# --
# IO

dataloaders = make_mnist_dataloaders(train_size=0.9, mode='fashion_mnist', 
    download=True, pretensor=True, seed=np.random.choice(1000))

# --
# Define environment

workers = {
    0 : StandardNet(lr=1e-3, dropout=[0.5, 0.5], kernel_size=[3, 3]).cuda(),
    1 : StandardNet(lr=1e-3, dropout=[0.25, 0.25], kernel_size=[5, 5]).cuda(),
}
env = PathEnv(workers=workers, dataloaders=dataloaders, seed=123)


ppo = SinglePathPPO(
    n_outputs=len(workers[0].layers),
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
    action_dim=len(workers[0].layers),
    mode='lever'
)

# burn in statistics
roll_gen.next()

# --
# Run

start_time = time()
ppo_epoch = 0
history = []
while roll_gen.step_index < args.total_steps:
    
    # --
    # Sample a batch of rollouts
    
    roll_gen.next()
    
    mean_reward = np.mean(list(env.acc_buffer)[-100:])
    action_counts = roll_gen.batch['actions'].cpu().numpy().sum(axis=0)
    
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

