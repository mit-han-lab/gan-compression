#!/usr/bin/env bash
python search.py --dataroot database/coco_stuff --dataset_mode coco \
  --model spade --netG super_mobile_spade \
  --restore_G_path logs/gaugan/coco_lite/supernet/checkpoints/latest_net_G.pth \
  --output_path logs/gaugan/coco_lite/supernet/results.pkl \
  --batch_size 32 --config_set channels-64 \
  --real_stat_path real_stat/coco_A.npz \
  --direction BtoA --max_dataset_size 200 \
  --budget 40e9 --ngf 64 --num_upsampling_layers normal --no_fid