#!/usr/bin/env bash
python train_supernet.py --dataroot database/cityscapes-origin \
  --supernet spade \
  --log_dir logs/gaugan/cityscapes/finetune \
  --restore_teacher_G_path logs/gaugan/cityscapes/mobile/export/latest_net_G.pth \
  --restore_student_G_path logs/gaugan/cityscapes/supernet/checkpoints/latest_net_G.pth \
  --restore_D_path logs/gaugan/cityscapes/supernet/checkpoints/latest_net_D.pth \
  --real_stat_path real_stat/cityscapes_A.npz \
  --drn_path drn-d-105_ms_cityscapes.pth \
  --cityscapes_path database/cityscapes-origin \
  --table_path datasets/table.txt \
  --gpu_ids 0,1,2,3,4,5,6,7 \
  --load_in_memory --no_fid \
  --config_str $1
