#!/usr/bin/env bash
python test.py --dataroot database/edges2shoes-r \
  --results_dir results-pretrained/pix2pix/edges2shoes-r/finetune \
  --ngf 48 --netG super_mobile_resnet_9blocks \
  --restore_G_path pretrained/pix2pix/edges2shoes-r/finetune/latest_net_G.pth \
  --config_str 32_32_48_32_48_48_16_16 \
  --real_stat_path real_stat/edges2shoes-r_B.npz \
  --need_profile --num_test 500
