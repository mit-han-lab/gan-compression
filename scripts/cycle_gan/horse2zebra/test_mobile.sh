#!/usr/bin/env bash
python test.py --dataroot database/horse2zebra/valA \
  --dataset_mode single \
  --results_dir results-pretrained/cycle_gan/horse2zebra/mobile \
  --ngf 64 --netG mobile_resnet_9blocks \
  --restore_G_path pretrained/cycle_gan/horse2zebra/mobile/latest_net_G.pth \
  --need_profile \
  --real_stat_path real_stat/horse2zebra_B.npz
