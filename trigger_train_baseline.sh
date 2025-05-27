#!/bin/bash

export WORLD_SIZE=4
export CUDA_VISIBLE_DEVICES=4,5,6,7

MASTER_PORT=29521
WORKERS_PER_GPU=12

torchrun --nproc_per_node=${WORLD_SIZE} --nnodes=1 --master_port=${MASTER_PORT} train.py \
--dataset laion_cogvlm \
--model ViT-B-32 \
--architecture ViT-B-32 \
--workers ${WORKERS_PER_GPU} \
--batch-size 200 \
--epochs 1 \
--expname "baseline" \
--freeze_only_vision \
--hard_negatives \
--hard_negatives_separate \
--shuffled_positive \
--hard_negative_freq 1 \
--additional_positives 2 \
--clip_loss_iterate \
--baseline \
--no_eval \
