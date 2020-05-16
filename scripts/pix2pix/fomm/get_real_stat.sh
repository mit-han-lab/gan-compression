#!/usr/bin/env bash
python3 get_real_stat.py \
  --input_nc 6 \
  --output_nc 3 \
  --dataroot database/fomm \
  --dataset_mode triplet \
  --output_path real_stat/fomm_B.npz
