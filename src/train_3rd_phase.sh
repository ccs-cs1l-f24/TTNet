#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=August


python main.py \
  --working-dir '../' \
  --saved_fn 'ttnet_3rd_phase' \
  --no_val \
  --batch_size 8 \
  --num_workers 4 \
  --lr 0.0001 \
  --lr_type 'step_lr' \
  --lr_step_size 10 \
  --lr_factor 0.2 \
  --gpu_idx 0 \
  --global_weight 1. \
  --no_seg \
  --event_weight 1. \
  --local_weight 1. \
  --pretrained_path ../checkpoints/ttnet_2nd_phase/ttnet_2nd_phase_epoch_30.pth \
  --smooth_labelling