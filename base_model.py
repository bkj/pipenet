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

class BaseModel(nn.Module):
    
    def __init__(self, loss_fn=F.cross_entropy):
        super(BaseModel, self).__init__()
        self.loss_fn = loss_fn
    
    def step(self, dataloaders, num_batches=np.inf):
        loader = dataloaders['train']
        _ = self.train()
        correct, total = 0, 0
        for batch_idx, (data, target) in enumerate(loader):
            data, target = Variable(data.cuda()), Variable(target.cuda())
            
            self.opt.zero_grad()
            output = self(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.opt.step()
            
            correct += (to_numpy(output).argmax(axis=1) == to_numpy(target)).sum()
            total += data.shape[0]
            
            if batch_idx > num_batches:
                break
        
        self.score['train'] = correct / total
        return self.score['train']
    
    def evaluate(self, dataloaders, mode='val', num_batches=np.inf):
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