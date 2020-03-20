#!/usr/bin/env bash
python test.py --dataroot database/maps \
--results_dir results-pretrained/pix2pix/map2sat/finetune \
--ngf 48 --netG super_mobile_resnet_9blocks \
--restore_G_path pretrained/pix2pix/map2sat/finetune/latest_net_G.pth \
--config_str 32_32_48_40_40_32_16_16 \
--real_stat_path real_stat/maps_A.npz \
--direction BtoA \
--need_profile --num_test 200
