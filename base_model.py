#!/usr/bin/env python

"""
    base_model.py
"""

from __future__ import print_function, division

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from helpers import to_numpy
from lr import LRSchedule

class BaseModel(nn.Module):
    
    def __init__(self, loss_fn=F.cross_entropy):
        super(BaseModel, self).__init__()
        self.loss_fn = loss_fn
        self.opt = None
        self.progress = 0
    
    def init_optimizer(self, opt, params, lr_scheduler, **kwargs):
        assert 'lr' not in kwargs, "BaseModel.init_optimizer: can't set LR from outside"
        self.opt = opt(params, lr=lr_scheduler(self.progress), **kwargs)
    
    def set_progress(self, progress):
        self.progress = progress
        LRSchedule.set_lr(self.opt, self.lr_scheduler(progress))
    
    def train_batch(self, data, target):
        _ = self.train()
        self.opt.zero_grad()
        output = self(data)
        loss = self.loss_fn(output, target)
        loss.backward()
        self.opt.step()
        return output
    
    def eval_batch(self, data):
        _ = self.eval()
        output = self(data)
        return output
    
    def train_epoch(self, dataloaders, num_batches=np.inf):
        assert self.opt is not None, "BaseModel: self.opt is None"
        
        loader = dataloaders['train']
        correct, total = 0, 0
        for batch_idx, (data, target) in enumerate(loader):
            data, target = Variable(data.cuda()), Variable(target.cuda())
            
            output = self.train_batch(data, target)
            
            correct += (to_numpy(output).argmax(axis=1) == to_numpy(target)).sum()
            total += data.shape[0]
            
            if batch_idx > num_batches:
                break
        
        self.score['train'] = correct / total
        return self.score['train']
    
    def eval_epoch(self, dataloaders, mode='val', num_batches=np.inf):
        assert self.opt is not None, "BaseModel: self.opt is None"
        
        loader = dataloaders[mode]
        if loader is None:
            return None
        else:
            _ = self.eval()
            correct, total = 0, 0 
            for batch_idx, (data, target) in enumerate(loader):
                data = Variable(data.cuda(), volatile=True)
                
                output = self(data)
                
                correct += (to_numpy(output).argmax(axis=1) == to_numpy(target)).sum()
                total += data.shape[0]
                
                if batch_idx > num_batches:
                    break
                
            self.score[mode] = correct / total
            return self.score[mode]