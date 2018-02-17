#!/bin/bash

# run.sh

# --
# Pretrained

export CUDA_VISIBLE_DEVICES=1

# ~0.94 accuracy
python pretrain.py \
    --train-size 0.9 \
    --seed 123 \
    --epochs 210 \
    --lr-schedule sgdr \
    --sgdr-period-length 30 \
    --sgdr-t-mult 2 \
    --outpath ./models/sgdr-train0.9

# --
# Pretrained model / default mask

# ~0.94 accuracy from first step
python main.py \
    --pretrained-weights ./models/sgdr-train0.9/weights \
    --seed 123 \
    --horizon \
    --outpath ./results/pretrained-fixed


# --
# Pretrained model / find the right mask

# w/ horizon
# 0.94 in 25 epochs
python main.py \
    --pretrained-weights ./models/sgdr-train0.9/weights \
    --seed 123 \
    --learn-mask \
    --ppo-epochs 50 \
    --outpath ./results/pretrained-learned \
    --horizon


# w/o horizon
# This means that rewards are spread across the last few actions
# Maybe this makes sense if you're training a model?  But maybe adds too much noise.
# 0.76 in 50 epochs
python main.py \
    --pretrained-weights ./models/sgdr-train0.9/weights \
    --seed 123 \
    --learn-mask \
    --ppo-epochs 50 \
    --outpath ./results/pretrained-learned-nohorizon

# --
# Dummy Train / learn mask

# ~ 0.8 in 50 epochs
python main.py \
    --seed 123 \
    --learn-mask \
    --ppo-epochs 50 \
    --outpath ./results/trained-learned-nohorizon \
    --horizon

# --
# Learned pipes / learned mask / constant learning rate

# ~ 0.90 in 100 epochs
run_name=0
python main.py \
    --seed 123 \
    --ppo-epochs 250 \
    --learn-mask \
    --train-mask \
    --outpath ./results/pipenet.$run_name \
    --horizon \
    --child-lr-init 0.01


# --
# Learned pipes / learned mask / SGDR

i=0
CUDA_VISIBLE_DEVICES=0 python main.py \
    --cuda \
    --seed 123 \
    --ppo-epochs 210 \
    --learn-mask \
    --train-mask \
    --outpath ./results/pipenet-sgdr.$i \
    --horizon \
    --child-lr-schedule sgdr \
    --child-lr-init 0.01 \
    --child-sgdr-period-length 30 \
    --child-sgdr-t-mult 2


i=1
CUDA_VISIBLE_DEVICES=0 python main.py \
    --cuda \
    --seed 123 \
    --ppo-epochs 210 \
    --learn-mask \
    --train-mask \
    --outpath ./results/pipenet-sgdr.$i \
    --horizon \
    --child-lr-schedule sgdr \
    --child-lr-init 0.01 \
    --child-sgdr-period-length 30 \
    --child-sgdr-t-mult 1

# ===========
# Pretraining

python pretrain.py \
    --train-size 0.9 \
    --seed 123 \
    --epochs 210 \
    --lr-schedule sgdr \
    --sgdr-period-length 30 \
    --sgdr-t-mult 2 \
    --outpath ./models/sgdr-train0.9

python pretrain.py \
    --train-size 0.9 \
    --seed 123 \
    --epochs 210 \
    --lr-schedule linear \
    --lr-init 0.05 \
    --outpath ./models/linear-lr0.05-train0.9

python pretrain.py \
    --train-size 1.0 \
    --seed 123 \
    --epochs 210 \
    --lr-schedule linear \
    --lr-init 0.05 \
    --outpath ./models/linear-lr0.05-train1.0

python pretrain.py \
    --train-size 1.0 \
    --seed 123 \
    --epochs 210 \
    --lr-schedule linear \
    --lr-init 0.10 \
    --outpath ./models/linear-lr0.10-train1.0

python pretrain.py \
    --train-size 1.0 \
    --seed 123 \
    --epochs 210 \
    --lr-schedule linear \
    --lr-init 0.10 \
    --outpath ./models/linear-lr0.10-train1.0-test

# !! These appear to be comparable to PCIFAR