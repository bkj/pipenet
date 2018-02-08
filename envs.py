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
from pipenet import PipeException

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
    def __init__(self, worker, dataloaders, seed=222):
        
        self.worker = worker
        self._pipes = np.array(list(worker.pipes.keys()))
        
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
    
    def step(self, actions):
        payout = []
        for action in actions:
            payout.append(self._step(action))
        
        payout = np.array(payout).reshape(-1, 1)
        
        is_done = np.array([False] * actions.shape[0]).reshape(-1, 1)
        state = self._random_state()
        
        return state, payout, is_done, None


class PretrainedEnv(BaseEnv):
    _eval_mode = 'val'
    
    def _step(self, mask):
        
        # --
        # Eval
        
        _ = self.worker.eval()
        
        data, target = next(self.dataloaders[self._eval_mode])
        data = Variable(data, volatile=True).cuda()
        
        mask_pipes = [tuple(pipe) for pipe in self._pipes[mask == 1]]
        self.worker.set_pipes(mask_pipes)
        
        try:
            output = self.worker(data)
            return (to_numpy(output).argmax(axis=1) == to_numpy(target)).mean()
        except PipeException:
            return 0.0
        except:
            raise


# class DummyTrainEnv(BaseEnv):
#     _eval_mode = 'val'
    
#     def _step(self, mask):
        
#         # --
#         # Train (using default pipes)
        
#         self.worker.reset_pipes()
#         data, target = next(self.dataloaders['train'])
#         data, target = Variable(data.cuda()), Variable(target.cuda())
#         _ = self.worker.train_batch(data, target)
        
#         # --
#         # Eval
        
#         _ = self.worker.eval()
        
#         data, target = next(self.dataloaders[self._eval_mode])
#         data = Variable(data, volatile=True).cuda()
        
#         mask_pipes = [tuple(pipe) for pipe in self._pipes[mask == 1]]
#         self.worker.set_pipes(mask_pipes)
        
#         try:
#             output = self.worker.eval_batch(data)
#             return (to_numpy(output).argmax(axis=1) == to_numpy(target)).mean()
#         except PipeException:
#             return 0.0
#         except:
#             raise
