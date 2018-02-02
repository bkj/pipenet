#!/usr/bin/env python

"""
    pbt.py
"""

from __future__ import print_function, division

import sys
import copy
import numpy as np
import pandas as pd

import torch
from torch import nn

from rsub import *
from matplotlib import pyplot as plt

from base_model import BaseModel
from data import make_mnist_dataloaders
from helpers import set_seeds, Flatten

# --
# Model

class Worker(BaseModel):
    def __init__(self, worker_id, knobs):
        super(Worker, self).__init__()
        
        self.worker_id = worker_id
        self.knobs = knobs
        self.layers = nn.Sequential(*[
            nn.Conv2d(1, 32, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.Dropout2d(p=knobs['dropout_p']),
            nn.MaxPool2d(2),
            nn.ReLU(),
            Flatten(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(p=knobs['dropout_p']),
            nn.Linear(128, 10),
        ])
        
        self.opt = torch.optim.Adam(
            self.parameters(),
            lr=10 ** knobs['lr'],
            weight_decay=10 ** knobs['weight_decay'],
        )
        
        self.score = {}
        
        self.can_explore = False
    
    def forward(self, x):
        return self.layers(x)
    
    def exploit(self, population, mode='val'):
        random_worker = np.random.choice(population.values())
        while random_worker.worker_id == self.worker_id:
            random_worker = np.random.choice(population.values())
        
        if random_worker.score[mode] > self.score[mode]:
            print('%d exploits %d' % (self.worker_id, random_worker.worker_id), file=sys.stderr)
            new_worker = Worker.clone(worker_id=self.worker_id, old_worker=random_worker)
            new_worker.can_explore = True
            return new_worker
        else:
            print('%d persists' % self.worker_id, file=sys.stderr)
            return self
    
    def explore(self):
        if self.can_explore:
            print('%d explores' % self.worker_id, file=sys.stderr)
            print(self.knobs, file=sys.stderr)
            self.knobs = {
                "lr"           : np.clip(self.knobs['lr'] * np.random.uniform(0.8, 1.2), -100, 0),
                "dropout_p"    : np.clip(self.knobs['dropout_p'] * np.random.uniform(0.8, 1.2), 0, 1),
                "weight_decay" : np.clip(self.knobs['weight_decay'] * np.random.uniform(0.8, 1.2), -100, 0),
            }
            print(self.knobs, file=sys.stderr)
            
            for param_group in self.opt.param_groups:
                param_group['lr'] = 10 ** self.knobs['lr']
                param_group['weight_decay'] = 10 ** self.knobs['weight_decay']
            
            self.layers[4].p = self.knobs['dropout_p']
            self.layers[10].p = self.knobs['dropout_p']
        else:
            raise Exception('cannot explore!')
        
        self.can_explore = False
    
    @classmethod
    def clone(cls, worker_id, old_worker):
        new_worker = cls(worker_id=worker_id, knobs=copy.deepcopy(old_worker.knobs))
        new_worker.load_state_dict(copy.deepcopy(old_worker.state_dict()))
        new_worker.opt.load_state_dict(copy.deepcopy(old_worker.opt.state_dict()))
        return new_worker


def random_knob():
    return {
        "lr"           : float(np.random.choice((-4, -3, -2))),
        "dropout_p"    : float(np.random.choice((0.4, 0.5, 0.6))),
        "weight_decay" : float(np.random.choice((-8, -7, -6))),
    }

# --
# Params

set_seeds(123)

pop_size = 10
num_phases = 10
num_epochs_per_phase = 10
total_steps = pop_size * num_phases * num_epochs_per_phase

dataloaders = make_mnist_dataloaders(train_size=0.9)

# --
# Run

population = dict([(worker_id, Worker(worker_id=worker_id, knobs=random_knob())) for worker_id in range(pop_size)])

all_res = []

for phase in range(num_phases):
    print(('\nphase %d  ' % phase + '+' * 50), file=sys.stderr)
    
    # Train workers for `num_epochs_per_phase` steps
    for worker_id, worker in population.items():
        worker = worker.cuda()
        
        print('-' * 5, file=sys.stderr)
        for epoch in range(num_epochs_per_phase):
            train_score = worker.step(dataloaders)
            res = {
                "phase"       : phase,
                "worker_id"   : worker_id,
                "epoch"       : epoch,
                "train_score" : train_score,
                "val_score"   : worker.evaluate(dataloaders, mode='val'),
                "test_score"  : worker.evaluate(dataloaders, mode='test'),
            }
            res.update(worker.knobs)
            all_res.append(res)
            print(all_res[-1])
        
        worker = worker.cpu()
    
    # Exploit and explore
    print('=' * 25, file=sys.stderr)
    new_population = {}
    for worker_id, worker in population.items():
        new_worker = worker.exploit(population)
        if new_worker.can_explore:
            new_worker.explore()
        
        new_population[worker_id] = new_worker
    
    population = new_population


# --
# Inspect


# df = pd.DataFrame(all_res)

# # train_scores = np.array(df.train_score)
# val_scores = np.array(df.val_score)
# test_scores = np.array(df.test_score)

# offset = np.array(num_epochs_per_phase * df.phase + df.epoch)

# _ = plt.scatter(offset, np.array(df.lr), alpha=0.5, s=5)
# show_plot()

# _ = plt.scatter(offset, np.array(df.weight_decay), alpha=0.5, s=5)
# show_plot()

# _ = plt.scatter(offset, np.array(df.dropout_p), alpha=0.5, s=5)
# show_plot()

# # _ = plt.scatter(offset, train_scores, alpha=0.25, s=3)
# _ = plt.scatter(offset, test_scores, alpha=0.25, s=3)
# _ = plt.scatter(offset, val_scores, alpha=0.25, s=3)
# _ = plt.ylim(0.95, 1.0)
# show_plot()

# standard_results = pd.read_csv('./standard_results.tsv', sep='\t')

# _ = plt.scatter(offset, test_scores, alpha=0.25, s=3)
# _ = plt.scatter(offset, val_scores, alpha=0.25, s=3)
# _ = plt.plot(standard_results.epoch, standard_results.test_score, c='red')
# _ = plt.ylim(0.95, 1.0)
# show_plot()

