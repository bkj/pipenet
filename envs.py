#!/usr/bin/env python

"""
    envs.py
"""

import numpy as np
from collections import Counter

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from helpers import to_numpy

# --
# Helpers

class Looper(object):
    def __init__(self, gen):
        self.epoch_batches = 0
        self.total_batches = 0
        self.epochs        = 0
        
        self._loop = self.__make_loop(gen)
    
    def __make_loop(self, gen):
        while True:
            self.epoch_batches = 0
            for x in gen:
                yield x
                self.total_batches += 1
                self.epoch_batches += 1
            
            self.epochs += 1
    
    def __next__(self):
        return next(self._loop)

# --
# Environments

class BaseEnv(object):
    _eval_mode = 'val'
    
    def __init__(self, worker=None, dataloaders=None, seed=222, learn_mask=True, train_mask=False, horizon=False):
        
        assert worker is not None
        assert dataloaders is not None
        
        self.worker = worker
        self._pipes = np.array(list(worker.pipes.keys()))
        
        self.learn_mask = learn_mask
        self.train_mask = train_mask
        self.horizon = horizon
        
        self.dataloaders = {
            "train" : Looper(dataloaders['train']),
            "val"   : Looper(dataloaders['val']),
            "test"  : Looper(dataloaders['test']),
        }
        
        self.rs = np.random.RandomState(seed=seed)
    
    def _random_state(self):
        return self.rs.normal(0, 1, (8, 32))
    
    def reset(self):
        return self._random_state()
    
    def step(self, masks):
        payout = []
        for mask in masks:
            payout.append(self._eval(mask))
        
        payout = np.array(payout).reshape(-1, 1)
        
        is_done = np.array([self.horizon] * masks.shape[0]).reshape(-1, 1)
        state = self._random_state()
        
        return state, payout, is_done, None
    
    def _eval(self, mask):
        data, target = next(self.dataloaders[self._eval_mode])
        data = Variable(data, volatile=True).cuda()
        
        if self.learn_mask:
            mask_pipes = [tuple(pipe) for pipe in self._pipes[mask == 1]]
            self.worker.set_pipes(mask_pipes)
        else:
            self.worker.reset_pipes()
        
        if self.worker.is_valid:
            return self.worker.eval_batch(data, target)
        else:
            return 0.0
    
    def _train(self, mask):
        data, target = next(self.dataloaders['train'])
        data, target = Variable(data.cuda()), Variable(target.cuda())
        
        if self.train_mask:
            mask_pipes = [tuple(pipe) for pipe in self._pipes[mask == 1]]
            self.worker.set_pipes(mask_pipes)
        else:
            self.worker.reset_pipes()
        
        if self.worker.is_valid:
            return self.worker.train_batch(data, target)
        else:
            return 0.0


class TrainEnv(BaseEnv):
    def step(self, masks):
        
        for mask in masks:
            print('train', mask)
            _ = self._train(mask)
        
        return super(TrainEnv, self).step(masks)

