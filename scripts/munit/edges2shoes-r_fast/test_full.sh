#!/usr/bin/env bash
# Inference without reference
python test.py --dataroot database/edges2shoes-r-unaligned/valA \
  --dataset_mode single \
  --results_dir results-pretrained/munit/edges2shoes-r_fast/full/no-style \
  --restore_G_A_path pretrained/munit/edges2shoes-r_fast/full/latest_net_G.pth \
  --need_profile --ngf 64 \
  --real_stat_path real_stat/edges2shoes-r_B.npz \
  --model munit_test --num_test 100 --netG munit

# Inference with references
python test.py --dataroot database/edges2shoes-r-unaligned/valA \
  --dataset_mode single \
  --results_dir results-pretrained/munit/edges2shoes-r_fast/full/ref1 \
  --restore_G_A_path pretrained/munit/edges2shoes-r_fast/full/latest_net_G.pth \
  --need_profile --ngf 64 \
  --real_stat_path real_stat/edges2shoes-r_B.npz \
  --model munit_test --num_test 100 --netG munit --ref_root imgs/edges2shoes/ref1
python test.py --dataroot database/edges2shoes-r-unaligned/valA \
  --dataset_mode single \
  --results_dir results-pretrained/munit/edges2shoes-r_fast/full/ref2 \
  --restore_G_A_path pretrained/munit/edges2shoes-r_fast/full/latest_net_G.pth \
  --need_profile --ngf 64 \
  --real_stat_path real_stat/edges2shoes-r_B.npz \
  --model munit_test --num_test 100 --netG munit --ref_root imgs/edges2shoes/ref2
