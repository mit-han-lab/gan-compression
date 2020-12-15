#!/usr/bin/env bash
python export.py \
  --input_path logs/cycle_gan/horse2zebra_fast/supernet/checkpoints/latest_net_G.pth \
  --output_path logs/cycle_gan/horse2zebra_fast/supernet/latest_net_G.pth \
  --ngf 64 --config_str $1
