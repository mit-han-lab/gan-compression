#!/usr/bin/env bash
python test.py --dataroot database/maps \
--results_dir results-pretrained/pix2pix/map2sat/distill \
--ngf 48 --netG mobile_resnet_9blocks \
--restore_G_path pretrained/pix2pix/map2sat/distill/latest_net_G.pth \
--real_stat_path real_stat/maps_A.npz \
--direction BtoA \
--need_profile --num_test 200
