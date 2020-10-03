#!/usr/bin/env bash
python test.py --dataroot database/coco_stuff \
  --model spade --dataset_mode coco \
  --results_dir results-pretrained/gaugan/coco_lite/compressed \
  --ngf 64 --netG sub_mobile_spade \
  --restore_G_path pretrained/gaugan/coco_lite/compressed/latest_net_G.pth \
  --real_stat_path real_stat/coco_A.npz --table_path datasets/table.txt \
  --need_profile --num_upsampling_layers normal --config_str 40_32_32_40_32_40_24_24
