#!/usr/bin/env bash
python train_supernet.py --dataroot database/maps \
--supernet resnet \
--log_dir logs/pix2pix/map2sat/supernet \
--teacher_ngf 96 \
--nepochs 200 --nepochs_decay 400 \
--save_epoch_freq 50 --save_latest_freq 20000 \
--eval_batch_size 16 \
--restore_teacher_G_path logs/pix2pix/map2sat/mobile/checkpoints/latest_net_G.pth \
--restore_student_G_path logs/pix2pix/map2sat/distill/checkpoints/latest_net_G.pth \
--restore_D_path logs/pix2pix/map2sat/distill/checkpoints/latest_net_D.pth \
--real_stat_path real_stat/maps_A.npz \
--direction BtoA --config_set channels-48 \
--lambda_recon 10 --lambda_distill 0.01
