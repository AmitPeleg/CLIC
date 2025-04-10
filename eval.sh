#!/bin/bash


export CUDA_VISIBLE_DEVICES=0

# architecture=ViT-B-32 ViT-B-16 ViT-L-14 ViT-L-16 ViT-H-14 ViT-H-16
architecture=ViT-B-32

# modelName=HF-CLIC-ViT-L-224-CogVLM #CLIPA, ViT-B-32, ViT-B-16, ViT-L-14, EVA, con-CLIP, TripletCLIP
modelName=HF-CLIC-ViT-B-32-224-CogVLM

python3 -m eval.general_eval --model $modelName --architecture $architecture --batch-size 200 --workers 8 --evaluation_metric imagenet 

python3 -m eval.general_eval --model $modelName --architecture $architecture --batch-size 200 --workers 8 --evaluation_metric zero_shot_classification 

python3 -m eval.general_eval --model $modelName --architecture $architecture --batch-size 200 --workers 8 --evaluation_metric coco2017_retrival 

python3 -m eval.general_eval --model $modelName --architecture $architecture --batch-size 200 --workers 8 --evaluation_metric flickr30k_retrival 

python3 -m eval.general_eval --model $modelName --architecture $architecture --batch-size 200 --workers 8 --evaluation_metric sugarcrepe 

python3 -m eval.general_eval --model $modelName --architecture $architecture --batch-size 200 --workers 8 --evaluation_metric sugarcrepe_pp 

python3 -m eval.general_eval --model $modelName --architecture $architecture --batch-size 200 --workers 8 --evaluation_metric winoground 
