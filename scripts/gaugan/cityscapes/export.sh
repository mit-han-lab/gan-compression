#!/usr/bin/env bash
python export.py \
  --model mobile_spade --input_nc 35 \
  --input_path logs/gaugan/cityscapes/finetune/checkpoints/latest_net_G.pth \
  --output_path logs/gaugan/cityscapes/compressed/latest_net_G.pth \
  --config_str $1
