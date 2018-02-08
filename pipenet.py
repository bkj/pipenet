#!/usr/bin/env python

"""
    pipenet.py
"""

from __future__ import print_function, division

import itertools
from dask import get
from pprint import pprint

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from basenet import BaseNet

# --
# Model definition

class PipeException(Exception):
    pass

class Accumulate(nn.Module):
    def __init__(self, agg_fn=torch.mean, name='noname'):
        super(Accumulate, self).__init__()
        self.agg_fn = agg_fn
        self.name = name
    
    def forward(self, parts):
        parts = [part for part in parts if part is not None]
        if len(parts) == 0:
            return None
        else:
            return self.agg_fn(torch.stack(parts), dim=0)
    
    def __repr__(self):
        return 'Accumulate(%s)' % self.name


class PBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, active=True, verbose=False):
        super(PBlock, self).__init__()
        self.bn1   = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False))
        
        self.active    = active
        self.verbose   = verbose
        self.in_planes = in_planes
        self.planes    = planes
        self.stride    = stride
    
    def forward(self, x):
        if self.active and (x is not None):
            if self.verbose:
                print('run:', self)
            
            out = F.relu(self.bn1(x))
            shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
            out = self.conv1(out)
            out = self.conv2(F.relu(self.bn2(out)))
            return out + shortcut
        else:
            return None
    
    def __repr__(self):
        return 'PBlock(%d -> %d | stride=%d | active=%d)' % (self.in_planes, self.planes, self.stride, self.active)


class PipeNet(BaseNet):
    def __init__(self, block=PBlock, num_blocks=[2, 2, 2, 2], lr_scheduler=None, num_classes=10, **kwargs):
        super(PipeNet, self).__init__(**kwargs)
        
        assert lr_scheduler is not None, "PipeNet.__init__: lr_scheduler is None"
        
        # --
        # Preprocessing
        
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        
        # --
        # Construct graph
        
        cell_sizes = [2 ** 6, 2 ** 7, 2 ** 8, 2 ** 9]
        
        self.cells = {}
        for cell_size in cell_sizes:
            self.cells[cell_size] = block(cell_size, cell_size, stride=1)
        
        self.pipes = {}
        for cell_size_0, cell_size_1 in itertools.combinations(cell_sizes, 2):
            self.pipes[(cell_size_0, cell_size_1)] = block(cell_size_0, cell_size_1, stride=int(cell_size_1 / cell_size_0))
        
        for k, v in self.cells.items():
            self.add_module(str(k), v)
        
        for k, v in self.pipes.items():
            self.add_module(str(k), v)
        
        # --
        # Classifier
        
        self.linear = nn.Linear(512, num_classes)
        
        # --
        # Set default pipes
        
        self.default_pipes = [(64, 128), (128, 256), (256, 512)]
        self.reset_pipes()
        
        self.init_optimizer(
            opt=torch.optim.SGD,
            lr_scheduler=lr_scheduler,
            params=self.parameters(),
            momentum=0.9,
            weight_decay=5e-4
        )
    
    def reset_pipes(self):
        self.set_pipes(self.default_pipes)
    
    def set_pipes(self, pipes):
        self.active_pipes = [tuple(pipe) for pipe in pipes]
        
        # Turn all pipes off
        for pipe in self.pipes.values():
            pipe.active = False
        
        # Turn active ones back on
        for pipe in self.active_pipes:
            self.pipes[pipe].active = True
        
        # Filter edges in graph
        self.graph = {
            # First cell
            'graph_input' : None,
            
            64         : (self.cells[64], 'graph_input'),
            
            (64, 128)  : (self.pipes[(64, 128)], 64),
            '128_acc'  : (Accumulate(name='128_acc'), self._filter_pipes([(64, 128)])),
            128        : (self.cells[128], '128_acc'),
            
            (64, 256)  : (self.pipes[(64, 256)], 64),
            (128, 256) : (self.pipes[(128, 256)], 128),
            '256_acc'  : (Accumulate(name='256_acc'), self._filter_pipes([(64, 256), (128, 256)])),
            256        : (self.cells[256], '256_acc'),
            
            (64, 512)  : (self.pipes[(64, 512)], 64),
            (128, 512) : (self.pipes[(128, 512)], 128),
            (256, 512) : (self.pipes[(256, 512)], 256),
            '512_acc'  : (Accumulate(name='512_acc'), self._filter_pipes([(64, 512), (128, 512), (256, 512)])),
            512        : (self.cells[512], '512_acc'),
        }
    
    def _filter_pipes(self, pipes):
        active_pipe_names = [pipe_name for pipe_name,pipe in self.pipes.items() if pipe.active]
        return [pipe for pipe in pipes if pipe in active_pipe_names]
    
    def forward(self, x, layer=512):
        out = F.relu(self.bn1(self.conv1(x)))
        
        self.graph['graph_input'] = out
        out = get(self.graph, layer)
        self.graph['graph_input'] = None
        
        if out is None:
            raise PipeException
        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


if __name__ == "__main__":
    
    model = PipeNet()
    print(model)
    
    x = Variable(torch.randn(16, 3, 32, 32))
    
    model.set_pipes([(64, 128), (128, 256), (256, 512)])
    print(model(x).shape)
