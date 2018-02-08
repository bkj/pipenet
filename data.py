#!/usr/bin/env python

"""
    data.py
"""

import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torchvision

def make_mnist_dataloaders(root='data', mode='mnist', train_size=1.0, train_batch_size=256, 
    eval_batch_size=256, num_workers=2, seed=123123, download=False, pretensor=False):
    
    if mode == 'mnist':
        if pretensor:
            transform = torchvision.transforms.Compose([
               torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])        
        else:
            transform = torchvision.transforms.Compose([
               torchvision.transforms.ToTensor(),
               torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
        
        trainset = torchvision.datasets.MNIST(root='%s/mnist' % root, train=True, 
            download=download, transform=transform, pretensor=pretensor)
        testset = torchvision.datasets.MNIST(root='%s/mnist' % root, train=False, 
            download=download, transform=transform, pretensor=pretensor)
    
    elif mode == 'fashion_mnist':
        if pretensor:
            transform = None
        else:
            transform = torchvision.transforms.Compose([
               torchvision.transforms.ToTensor(),
            ])
        
        trainset = torchvision.datasets.FashionMNIST(root='%s/fashion_mnist' % root, train=True, 
            download=download, transform=transform, pretensor=pretensor)
        testset = torchvision.datasets.FashionMNIST(root='%s/fashion_mnist' % root, train=False, 
            download=download, transform=transform, pretensor=pretensor)
    
    else:
        raise Exception
    
    return _make_loaders(trainset, testset, train_size, train_batch_size, eval_batch_size, num_workers, seed)


def make_cifar_dataloaders(root='data', mode='CIFAR10', train_size=1.0, train_batch_size=128, 
    eval_batch_size=128, num_workers=2, seed=123123, download=False):
    
    if mode == 'CIFAR10':
        
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        trainset = torchvision.datasets.CIFAR10(root='%s/CIFAR10' % root, train=True, download=download, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='%s/CIFAR10' % root, train=False, download=download, transform=transform_test)
    else:
        raise Exception
    
    return _make_loaders(trainset, testset, train_size, train_batch_size, eval_batch_size, num_workers, seed)


def _make_loaders(trainset, testset, train_size, train_batch_size, eval_batch_size, num_workers, seed):
    if train_size < 1:
        train_inds, val_inds = train_test_split(np.arange(len(trainset)), train_size=train_size, random_state=seed)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=train_batch_size, num_workers=num_workers, pin_memory=True,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(train_inds),
        )
        
        valloader = torch.utils.data.DataLoader(
            trainset, batch_size=eval_batch_size, num_workers=num_workers, pin_memory=True,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(val_inds),
        )
    else:
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=train_batch_size, num_workers=num_workers, pin_memory=True,
            shuffle=True,
        )
        
        valloader = None
    
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=eval_batch_size, num_workers=num_workers, pin_memory=True,
        shuffle=False,
    )
    
    return {
        "train" : trainloader,
        "test" : testloader,
        "val" : valloader,
    }
