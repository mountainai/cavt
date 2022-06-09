#!/usr/bin/env bash
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
/home/myuser/.conda/envs/tensorflow/bin/python  tools/test.py \
optimal_checkpoint/cait_tiny_engwild2020_d12h16_d4444/cait_tiny_engwild2020.py  \
optimal_checkpoint/cait_tiny_engwild2020_d12h16_d4444/best_mse_epoch_18.pth  --eval mse --seed 35

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
/home/myuser/.conda/envs/tensorflow/bin/python  tools/test.py \
optimal_checkpoint/cait_tiny_daisee_d12h16_d3333/cait_tiny_daisee.py  \
optimal_checkpoint/cait_tiny_daisee_d12h16_d3333/best_mse_epoch_11.pth  --eval mse --seed 10
