#!/usr/bin/env bash
python train.py --dataroot database/cityscapes-origin \
  --model spade --dataset_mode cityscapes \
  --log_dir logs/gaugan/cityscapes/mobile \
  --netG mobile_spade \
  --real_stat_path real_stat/cityscapes_A.npz \
  --drn_path drn-d-105_ms_cityscapes.pth \
  --cityscapes_path database/cityscapes-origin \
  --table_path datasets/table.txt \
  --gpu_ids 0,1,2,3,4,5,6,7 --load_in_memory --no_fid \
  --norm_G spectralspadesyncbatch3x3

# remove the spectral normalization for the distillation and once-for-all network training
python remove_spectral_norm.py \
  --restore_G_path logs/gaugan/cityscapes/mobile/checkpoints/latest_net_G.pth \
  --output_path logs/gaugan/cityscapes/mobile/export/latest_net_G.pth
