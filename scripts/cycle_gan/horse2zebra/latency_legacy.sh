#!/usr/bin/env bash
python latency.py --dataroot database/horse2zebra/valA \
  --dataset_mode single \
  --results_dir results-pretrained/cycle_gan/horse2zebra/legacy \
  --config_str 24_24_24_32_20_16 \
  --restore_G_path pretrained/cycle_gan/horse2zebra/legacy/latest_net_G.pth \
  --need_profile \
  --real_stat_path real_stat/horse2zebra_B.npz \
  --netG legacy_sub_mobile_resnet_9blocks
