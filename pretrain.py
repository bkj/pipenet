#!/usr/bin/env python

"""
    pretrain.py
    
    !! Gives exact same results as pcifar trainer if:
        - num_workers = 0
        - random augmentation turned off
        - cudnn.deterministic = True
"""

from __future__ import print_function, division

import os
import sys
import json
import argparse
import numpy as np
from time import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from helpers import set_seeds, to_numpy
from data import make_cifar_dataloaders
from pipenet import PipeNet
from lr import LRSchedule

torch.backends.cudnn.benchmark = True

# --
# Params

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outpath', type=str)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr-schedule', type=str, default='linear')
    parser.add_argument('--lr-init', type=float, default=0.05)
    
    # SGDR options
    parser.add_argument('--sgdr-period-length', type=float, default=30)
    parser.add_argument('--sgdr-t-mult',  type=float, default=2)
    
    parser.add_argument('--train-size', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()

# --
# Args

if __name__ == "__main__":
    args = parse_args()
    
    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)
    
    json.dump(vars(args), open(os.path.join(args.outpath, 'config'), 'w'))
    
    set_seeds(args.seed)
    
    dataloaders = make_cifar_dataloaders(train_size=args.train_size, download=False, seed=args.seed)
    lr_scheduler = getattr(LRSchedule, args.lr_schedule)(**{
        "lr_init"       : args.lr_init,
        "epochs"        : args.epochs,
        "period_length" : args.sgdr_period_length,
        "t_mult"        : args.sgdr_t_mult,
    })
    
    worker = PipeNet(lr_scheduler=lr_scheduler, loss_fn=F.cross_entropy).cuda()
    worker.reset_pipes()
    print(worker, file=sys.stderr)
    
    logfile = open(os.path.join(args.outpath, 'log'), 'w')
    
    for epoch in range(args.epochs):
        t = time()
        train_score = worker.train_epoch(dataloaders)
        res = {
            "epoch"        : epoch,
            "lr"           : worker.lr,
            "train_score"  : train_score,
            "val_score"    : worker.eval_epoch(dataloaders, mode='val'),
            "test_score"   : worker.eval_epoch(dataloaders, mode='test'),
            "time"         : time() - t,
        }
        
        print(json.dumps(res))
        print(json.dumps(res), file=logfile)
    
    logfile.close()
    
    model_path = os.path.join(args.outpath, 'weights')
    print('saving model: %s' % model_path, file=sys.stderr)
    torch.save(worker.state_dict(), model_path)

