#!/bin/bash -eux

python main.py \
    --optimization Sequential_BitWidth \
    --num-trials 5 \
    --seed-opt 0 \
    --seed-merge 0 \
    --wandb False \
    --wandb-project 'MergeQuant' \
    --result-dir 'results' \
    --figure-dir 'figures' \
    --test-best True \
    --create-graph True \

