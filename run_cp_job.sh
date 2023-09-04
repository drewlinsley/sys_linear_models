#!/usr/bin/env bash

# Run one worker who can access the DB and run a job

read -p "Which GPU: " GPU
echo "You entered GPU: $GPU"

while :
do
    CUDA_VISIBLE_DEVICES=$GPU python train_cp_controls.py
done

