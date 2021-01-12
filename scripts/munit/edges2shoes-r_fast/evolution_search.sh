#!/usr/bin/env bash
python evolution_search.py --phase train --dataroot database/edges2shoes-r-unaligned/trainA \
  --dataset_mode single --netG super_mobile_munit --model munit_test \
  --restore_G_A_path logs/munit/edges2shoes-r_fast/supernet/checkpoints/latest_net_G.pth \
  --output_dir logs/munit/edges2shoes-r_fast/supernet/evolution \
  --ngf 64 --batch_size 32 --config_set channels-64 \
  --real_stat_path real_stat/edges2shoes-r-unaligned_subtrain_B.npz --budget 2.7e9 \
  --meta_path datasets/metas/edges2shoes-r-unaligned/train2A.meta --weighted_sample 2 --weight_strategy geometric
