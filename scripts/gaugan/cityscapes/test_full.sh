#!/usr/bin/env bash
python test.py --dataroot database/cityscapes-origin \
--model spade --dataset_mode cityscapes \
--results_dir results-pretrained/gaugan/cityscapes/full \
--ngf 64 --netG spade \
--restore_G_path pretrained/gaugan/cityscapes/full/latest_net_G.pth \
--real_stat_path real_stat/cityscapes_A.npz \
--drn_path drn-d-105_ms_cityscapes.pth \
--cityscapes_path database/cityscapes-origin \
--table_path datasets/table.txt --need_profile