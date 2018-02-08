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
from pipenet import AccumulateException

class BaseEnv(object):
    def __init__(self, worker, dataloaders, seed=123, cache_data=True):
        self.worker = worker
        self._pipes = np.array(list(worker.pipes.keys()))
        
        if cache_data:
            self.dataloaders = {
                "train" : self._iterwrapper(dataloaders['train'], mode='train'),
                "val"   : self._iterwrapper(dataloaders['val'], mode='val'),
                "test"  : self._iterwrapper(dataloaders['test'], mode='test'),
            }
        else:
            raise Exception('not implemented w/o cache_data')
        
        self.rs = np.random.RandomState(seed=seed)
        self.train = True
        self.epochs = Counter()
        self.steps = 0
    
    def _iterwrapper(self, loader, mode):
        data = list(iter(loader))
        X = torch.cat([t[0] for t in data])
        y = torch.cat([t[1] for t in data])
        
        while True:
            chunks = torch.chunk(torch.randperm(X.shape[0]), X.shape[0] // loader.batch_size)
            for chunk in chunks:
                yield X[chunk], y[chunk]
            
            self.epochs[mode] += 1
    
    def _random_state(self):
        return self.rs.normal(0, 1, (8, 32))
    
    def reset(self):
        return self._random_state()


class PretrainedEnv(BaseEnv):
    def _step(self, mask, mode='val'):
        
        # --
        # Eval
        
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
            payout.append(self._step(action, mode='test'))
        
        payout = np.array(payout).reshape(-1, 1)
        
        is_done = np.array([False] * actions.shape[0]).reshape(-1, 1)
        state = self._random_state()
        
        self.steps += 1
        
        return state, payout, is_done, None


class DummyTrainEnv(BaseEnv):
    def _step(self, mask, mode='val'):
        
        # --
        # Train (using default pipes)
        
        self.worker.reset_pipes()
        data, target = next(self.dataloaders['train'])
        data, target = Variable(data.cuda()), Variable(target.cuda())
        output = self.worker.train_batch(data, target)
        
        # --
        # Eval
        
        _ = self.worker.eval()
        
        data, target = next(self.dataloaders[mode])
        data = Variable(data, volatile=True).cuda()
        
        mask_pipes = [tuple(pipe) for pipe in self._pipes[mask == 1]]
        self.worker.set_pipes(mask_pipes)
        
        try:
            output = self.worker.eval_batch(data)
            return (to_numpy(output).argmax(axis=1) == to_numpy(target)).mean()
        except AccumulateException:
            return 0.0
        except:
            raise
    
    def step(self, actions):
        payout = []
        for action in actions:
            payout.append(self._step(action, mode='test'))
        
        payout = np.array(payout).reshape(-1, 1)
        
        is_done = np.array([False] * actions.shape[0]).reshape(-1, 1)
        state = self._random_state()
        
        self.steps += 1
        
        return state, payout, is_done, None

