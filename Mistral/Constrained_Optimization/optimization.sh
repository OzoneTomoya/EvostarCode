#!/bin/bash -eux

python main.py \
    --optimization Sequential_MergeParameter \
    --num-trials 5 \
    --seed-cmaes 0 \
    --seed-merge 0 \
    --wandb False \
    --wandb-project 'MergeQuant' \
    --result-dir 'results' \
    --figure-dir 'figures' \
    --save-interval 100 \
    --test-best True \
    --create-graph True \
    --target-bit 8 \