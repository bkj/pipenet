#!/usr/bin/env python

"""
    main.py
"""

from __future__ import print_function, division

import os
import sys
import pickle
import atexit
import argparse
import numpy as np
from time import time

import torch
from torch.nn import functional as F
from torch.autograd import Variable

# Plotting
from rsub import *
from matplotlib import pyplot as plt

# Pipenet
from helpers import set_seeds, to_numpy
from data import make_cifar_dataloaders
from pipenet import PipeNet
from envs import BaseEnv, TrainEnv
from lr import LRSchedule

# PPO
sys.path.append('/home/bjohnson/projects/simple_ppo')
from models.path import SinglePathPPO
from rollouts import RolloutGenerator

# --
# Params

def parse_args():
    """ parameters mimic openai/baselines """
    parser = argparse.ArgumentParser()
    
    # --
    # Pipenet options
    
    parser.add_argument('--outpath', type=str, default='last-history.pkl')
    parser.add_argument('--pretrained-weights', type=str)
    parser.add_argument('--learn-mask', action="store_true")
    parser.add_argument('--train-mask', action="store_true")
    
    parser.add_argument('--no-horizon', action="store_true") # infinite vs 0 horizon
    
    parser.add_argument('--child-lr-init', type=float, default=0.01)
    parser.add_argument('--child-lr-schedule', type=str, default='constant')
    parser.add_argument('--child-lr-epochs', type=int, default=1000) # !! _estimated_
    
    # --
    # PPO Options
    
    parser.add_argument('--ppo-epochs', type=int, default=int(1e5))
    parser.add_argument('--total-steps', type=int, default=int(40e6))
    parser.add_argument('--steps-per-batch', type=int, default=64)
    parser.add_argument('--epochs-per-batch', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--num-frames', type=int, default=4)
    
    parser.add_argument('--advantage-gamma', type=float, default=0.99)
    parser.add_argument('--advantage-lambda', type=float, default=0.95)
    
    parser.add_argument('--clip-eps', type=float, default=0.2)
    parser.add_argument('--ppo-eps', type=float, default=1e-5)
    parser.add_argument('--ppo-lr', type=float, default=1e-3)
    parser.add_argument('--entropy-penalty', type=float, default=0.001)
    
    # --
    # Additional options 
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--cuda', action="store_true")
    
    args = parser.parse_args()
    
    def save():
        pickle.dump(history, open(args.outpath, 'wb'))
        
    atexit.register(save)
    
    return args

# --
# Args

args = parse_args()
print(args)
set_seeds(args.seed)

# --
# IO

dataloaders = make_cifar_dataloaders(train_size=0.9, download=False, seed=args.seed)

# --
# Define environment

if args.pretrained_weights is not None:
    worker = PipeNet(loss_fn=F.cross_entropy).cuda()
    worker.load_state_dict(torch.load(args.pretrained_weights))
    env = BaseEnv(
        worker=worker,
        dataloaders=dataloaders,
        seed=args.seed,
        learn_mask=args.learn_mask,
        train_mask=args.train_mask,
        no_horizon=args.no_horizon,
    )
else:
    lr_scheduler = getattr(LRSchedule, args.child_lr_schedule)(
        lr_init=args.child_lr_init,
        epochs=args.child_lr_epochs,
    )
    worker = PipeNet(loss_fn=F.cross_entropy, lr_scheduler=lr_scheduler).cuda()
    env = TrainEnv(
        worker=worker,
        dataloaders=dataloaders,
        seed=args.seed,
        learn_mask=args.learn_mask,
        train_mask=args.train_mask,
        no_horizon=args.no_horizon,
    )

ppo = SinglePathPPO(
    n_outputs=len(worker.pipes),
    ppo_lr=args.ppo_lr,
    ppo_eps=args.ppo_eps,
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
        
        "train_epochs" : env.dataloaders['train'].epochs,
        "val_epochs"   : env.dataloaders['val'].epochs,
        "test_epochs"  : env.dataloaders['test'].epochs,
        
        "learn_mask" : args.learn_mask,
        "pretrained_weights" : args.pretrained_weights is not None,
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
    
    if ppo_epoch > args.ppo_epochs:
        break


# # --
# # Inspect


# worker.reset_pipes()

# worker.eval_epoch(dataloaders, mode='test')
# worker.eval_epoch(dataloaders, mode='val')
# worker.eval_epoch(dataloaders, mode='test')

# _ = plt.plot([h['mean_reward'] for h in history][:300])
# _ = plt.ylim(0.85, 0.95)
# show_plot()

# action_counts = np.vstack([h['action_counts'] for h in history])[:500]
# for i, r in enumerate(action_counts.T):
#     _ = plt.plot(r, alpha=0.5, label=i)

# plt.legend(loc='lower right')
# show_plot()
