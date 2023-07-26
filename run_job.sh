#!/usr/bin/env bash

# Run one worker who can access the DB and run a job

read -p "Which GPU: " GPU
echo "You entered GPU: $GPU"

CUDA_VISIBLE_DEVICES=$GPU accelerate launch train_and_eval_one_model.py

