#!/bin/bash

weights_list=(
  "0,1,0,0"
  "0,0,1,0"
  "0,0,0,1"
  "1,0,0,0"
)

for weights in "${weights_list[@]}"; do
  python train.py \
    --dataset_name=cifar10 \
    --total_kimg=200 \
    --batch=256 \
    --lr=5e-5 \
    --num_steps=4 \
    --M=3 \
    --afs=False \
    --sampler_tea=dpmpp \
    --max_order=3 \
    --predict_x0=True \
    --lower_order_final=True \
    --schedule_type=polynomial \
    --schedule_rho=7 \
    --use_step_condition=False \
    --is_second_stage=False \
    --use_repeats=true \
    --seed=1913901889 \
    --weight_ls=$weights
done
