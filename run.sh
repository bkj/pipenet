#!/bin/bash

# run.sh


export CUDA_VISIBLE_DEVICES=1

python pretrain.py --train-size 0.9 --seed 123 --outpath ./models/linear-0.9

python main.py --pretrained-weights ./models/linear-0.9/weights
