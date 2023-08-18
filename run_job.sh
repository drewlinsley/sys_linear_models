#!/usr/bin/env bash

# Run one worker who can access the DB and run a job

read -p "Which GPU: " GPU
echo "You entered GPU: $GPU"

while :
do
    CUDA_VISIBLE_DEVICES=$GPU accelerate launch --mixed_precision fp16 train_and_eval_one_model.py
    # CUDA_VISIBLE_DEVICES=$GPU python train_and_eval_one_model.py
done

