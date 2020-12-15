#!/usr/bin/env bash
python export.py \
  --model mobile_spade --input_nc 182 \
  --input_path logs/gaugan/coco_fast/finetune/checkpoints/latest_net_G.pth \
  --output_path logs/gaugan/coco_fast/compressed/latest_net_G.pth \
  --config_str $1 --ngf 64 --num_upsampling_layers normal --contain_dontcare_label