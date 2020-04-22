#!/usr/bin/env bash
python test.py --dataroot database/edges2shoes-r \
  --results_dir results-pretrained/pix2pix/edges2shoes-r/mobile \
  --ngf 64 --netG mobile_resnet_9blocks \
  --restore_G_path pretrained/pix2pix/edges2shoes-r/mobile/latest_net_G.pth \
  --real_stat_path real_stat/edges2shoes-r_B.npz \
  --need_profile --num_test 500
