#!/bin/bash

export WORLD_SIZE=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

MASTER_PORT=29522
WORKERS_PER_GPU=12

torchrun --nproc_per_node=${WORLD_SIZE} --nnodes=1 --master_port=${MASTER_PORT} train.py \
--dataset laion_cogvlm \
--model ViT-B-32 \
--architecture ViT-B-32 \
--workers ${WORKERS_PER_GPU} \
--batch-size 200 \
--epochs 1 \
--expname "negclip" \
--freeze_only_vision \
--hard_negatives \
--no_concat \

