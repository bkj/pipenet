#!/bin/bash

# run.sh

CUDA_VISIBLE_DEVICES=1 python experiments/pipenet-pretrained.py 

CUDA_VISIBLE_DEVICES=1 python experiments/pipenet-pretrained.py --pretrained-path ./models/pipenet-constant-20

CUDA_VISIBLE_DEVICES=1 python experiments/pipenet-pretrained.py --env dummy_train