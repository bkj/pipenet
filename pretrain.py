#!/usr/bin/env python

"""
    pretrain.py
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

# --
# Params

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outpath', type=str)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr-schedule', type=str, default='linear')
    parser.add_argument('--lr-init', type=float, default=0.1)
    parser.add_argument('--train-size', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()

# --
# Args

if __name__ == "__main__":
    args = parse_args()
    
    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)
    
    set_seeds(args.seed)
    print('seed=%d' % args.seed)
    
    dataloaders = make_cifar_dataloaders(train_size=0.9, download=False, seed=args.seed)
    lr_scheduler = getattr(LRSchedule, args.lr_schedule)(lr_init=args.lr_init, epochs=args.epochs)
    worker = PipeNet(lr_scheduler=lr_scheduler, loss_fn=F.cross_entropy).cuda()
    
    logfile = open(os.path.join(args.outpath, 'log'), 'w')
    
    for epoch in range(args.epochs):
        t = time()
        train_score = worker.train_epoch(dataloaders)
        res = {
            "epoch"        : epoch,
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

