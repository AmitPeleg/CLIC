#!/bin/bash


export CUDA_VISIBLE_DEVICES=5

architecture=ViT-B-32

modelName=ViT-B-32

python3 -m eval.general_eval --model $modelName --architecture $architecture --batch-size 200 --workers 8 --evaluation_metric imagenet

python3 -m eval.general_eval --model $modelName --architecture $architecture --batch-size 200 --workers 8 --evaluation_metric zero_shot_classification 

python3 -m eval.general_eval --model $modelName --architecture $architecture --batch-size 200 --workers 8 --evaluation_metric coco2017_retrival 

python3 -m eval.general_eval --model $modelName --architecture $architecture --batch-size 200 --workers 8 --evaluation_metric flickr30k_retrival 

python3 -m eval.general_eval --model $modelName --architecture $architecture --batch-size 200 --workers 8 --evaluation_metric sugarcrepe 

python3 -m eval.general_eval --model $modelName --architecture $architecture --batch-size 200 --workers 8 --evaluation_metric sugarcrepe_pp 

python3 -m eval.general_eval --model $modelName --architecture $architecture --batch-size 200 --workers 8 --evaluation_metric winoground
