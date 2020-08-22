#!/usr/bin/env bash
python latency.py --dataroot database/edges2shoes-r \
  --results_dir results-pretrained/pix2pix/edges2shoes-r/legacy \
  --restore_G_path pretrained/pix2pix/edges2shoes-r/legacy/latest_net_G.pth \
  --config_str 32_32_40_48_16_32 \
  --real_stat_path real_stat/edges2shoes-r_B.npz \
  --need_profile --num_test 500 \
  --netG legacy_sub_mobile_resnet_9blocks
