#!/usr/bin/env bash
python evolution_search.py --dataroot database/coco_stuff --dataset_mode coco \
  --model spade --netG super_mobile_spade \
  --restore_G_path logs/gaugan/coco_fast/supernet/checkpoints/latest_net_G.pth \
  --output_dir logs/gaugan/coco_fast/evolution \
  --batch_size 32 --config_set channels-64-coco \
  --real_stat_path real_stat/coco_A.npz \
  --direction BtoA --max_dataset_size 500 \
  --budget 36e9 --ngf 64 --num_upsampling_layers normal --no_fid --criterion mIoU