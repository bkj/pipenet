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

# --
# Learned pipes / learned mask (w/ one bad pipe)

for i in $(seq 3); do
    python main.py \
        --seed 123 \
        --ppo-epochs 400 \
        --learn-mask \
        --train-mask \
        --outpath ./results/pipenet-qblock.$i.pkl
done

# --
# Open questions

# !! How should the attribution business be working?
#   Should record number of times a cell is sucessfully trained


# =======================================================================================

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

