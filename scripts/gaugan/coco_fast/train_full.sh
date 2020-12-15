#!/usr/bin/env bash
python train.py --dataroot database/coco_stuff \
  --model spade --dataset_mode coco \
  --log_dir logs/gaugan/coco/full \
  --netG spade --batch_size 32 --nepochs 100 --nepochs_decay 0 \
  --real_stat_path real_stat/coco_A.npz \
  --table_path datasets/table.txt \
  --gpu_ids 0,1,2,3 \
  --norm_G spectralspadesyncbatch3x3 \
  --num_upsampling_layers normal \
  --save_epoch_freq 1

# remove the spectral normalization for the distillation and once-for-all network training
python remove_spectral_norm.py --netG spade \
  --restore_G_path logs/gaugan/coco/full/checkpoints/latest_net_G.pth \
  --output_path logs/gaugan/coco/full/export/latest_net_G.pth \
  --num_upsampling_layers normal \
  --input_nc 182 --contain_dontcare_label