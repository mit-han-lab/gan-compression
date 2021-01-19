#!/usr/bin/env bash
# Inference without reference
python test.py --dataroot database/edges2shoes-r-unaligned/valA \
  --dataset_mode single \
  --results_dir results-pretrained/munit/edges2shoes-r_fast/compressed/no-style \
  --ngf 64 --need_profile \
  --restore_G_A_path pretrained/munit/edges2shoes-r_fast/compressed/latest_net_G.pth \
  --real_stat_path real_stat/edges2shoes-r_B.npz \
  --model munit_test --num_test 100 --netG sub_mobile_munit \
  --config_str 16_16_16_24_56_16_40_40_32_24

# Inference with references
python test.py --dataroot database/edges2shoes-r-unaligned/valA \
  --dataset_mode single \
  --results_dir results-pretrained/munit/edges2shoes-r_fast/compressed/ref1 \
  --ngf 64 --need_profile \
  --restore_G_A_path pretrained/munit/edges2shoes-r_fast/compressed/latest_net_G.pth \
  --real_stat_path real_stat/edges2shoes-r_B.npz \
  --model munit_test --num_test 100 --netG sub_mobile_munit \
  --config_str 16_16_16_24_56_16_40_40_32_24 --ref_root imgs/edges2shoes/ref1
python test.py --dataroot database/edges2shoes-r-unaligned/valA \
  --dataset_mode single \
  --results_dir results-pretrained/munit/edges2shoes-r_fast/compressed/ref2 \
  --ngf 64 --need_profile \
  --restore_G_A_path pretrained/munit/edges2shoes-r_fast/compressed/latest_net_G.pth \
  --real_stat_path real_stat/edges2shoes-r_B.npz \
  --model munit_test --num_test 100 --netG sub_mobile_munit \
  --config_str 16_16_16_24_56_16_40_40_32_24 --ref_root imgs/edges2shoes/ref2
