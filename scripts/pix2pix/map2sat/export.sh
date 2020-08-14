#!/usr/bin/env bash
python export.py \
  --input_path logs/pix2pix/map2sat/finetune/checkpoints/latest_net_G.pth \
  --output_path logs/pix2pix/map2sat/compressed/latest_net_G.pth \
  --config_str $1
