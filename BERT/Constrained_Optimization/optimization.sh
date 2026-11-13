#!/bin/bash -eux

python main.py \
    --optimization Sequential_MergeParameter \
    --num-trials 5 \
    --seed-cmaes 0 \
    --seed-merge 0 \
    --wandb False \
    --wandb-project 'MergeQuant' \
    --num-save-trials 100 \
    --result-dir 'results' \
    --figure-dir 'figures' \
    --test-best True \
    --create-graph True \
    --target-bit 6 \

