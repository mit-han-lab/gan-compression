#!/usr/bin/env bash
python test.py --dataroot database/cityscapes \
--results_dir results-pretrained/pix2pix/cityscapes/finetune \
--ngf 48 --netG super_mobile_resnet_9blocks \
--config_str 32_32_40_48_48_48_24_24 \
--restore_G_path pretrained/pix2pix/cityscapes/finetune/latest_net_G.pth \
--real_stat_path real_stat/cityscapes_A.npz \
--drn_path drn-d-105_ms_cityscapes.pth \
--cityscapes_path database/cityscapes-origin \
--table_path datasets/table.txt \
--direction BtoA --need_profile