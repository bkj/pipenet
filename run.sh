#!/bin/bash

# run.sh

# --
# Pretrained

export CUDA_VISIBLE_DEVICES=1

python pretrain.py --train-size 0.9 --seed 123 --outpath ./models/linear-0.9

# --
# Pretrained / default mask
# High accuracy from beginning

python main.py --pretrained-weights ./models/linear-0.9/weights \
    --seed 123 \
    --horizon \
    --outpath ./results/pretrained-fixed.pkl

# --
# Pretrained / learn mask
# Low accuracy -> top accuracy

python main.py \
    --pretrained-weights ./models/linear-0.9/weights \
    --seed 123 \
    --learn-mask \
    --ppo-epochs 50 \
    --horizon \
    --outpath ./results/pretrained-learned.pkl


python main.py \
    --pretrained-weights ./models/linear-0.9/weights \
    --seed 123 \
    --learn-mask \
    --ppo-epochs 50 \
    --outpath ./results/pretrained-learned-nohorizon.pkl

# --
# Dummy Train / fixed mask

python main.py \
    --seed 123 \
    --ppo-epochs 50 \
    --outpath ./results/trained-fixed-nohorizon.pkl

# --
# Dummy Train / learn mask

python main.py \
    --seed 123 \
    --ppo-epochs 50 \
    --learn-mask \
    --outpath ./results/trained-learned-nohorizon.pkl

# --
# Learned pipes / learned mask

python main.py \
    --seed 123 \
    --ppo-epochs 50 \
    --learn-mask \
    --train-mask \
    --outpath ./results/pipenet.pkl

# This works -- converges to all 1s
# Is that a degenerate solution or the right one?

# --
# Learned pipes / learned mask (w/ one bad pipe)

python main.py \
    --seed 123 \
    --ppo-epochs 500 \
    --learn-mask \
    --train-mask \
    --outpath ./results/pipenet-qblock-3.pkl

# Number of train batches here is high, because some of the batches fail
# Should keep track of number of failed training pulls

# --
# Open questions

# !! How should the attribution business be working?
#   Should record number of times a cell is sucessfully trained