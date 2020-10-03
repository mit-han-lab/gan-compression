#!/usr/bin/env bash
python train_supernet.py --dataroot database/coco_stuff --dataset_mode coco \
  --supernet spade --teacher_netG spade --student_ngf 64 \
  --log_dir logs/gaugan/coco_lite/supernet \
  --restore_teacher_G_path pretrained/gaugan/coco_lite/full/latest_net_G.pth \
  --real_stat_path real_stat/coco_A.npz \
  --gpu_ids 0,1,2,3 \
  --nepochs 100 --nepochs_decay 0 \
  --config_set channels-64 \
  --num_upsampling_layers normal --batch_size 32 --save_epoch_freq 1
