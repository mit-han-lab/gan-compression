#!/usr/bin/env bash
python export.py \
  --input_path logs/pix2pix/cityscapes_fast/supernet/checkpoints/latest_net_G.pth \
  --output_path logs/pix2pix/cityscapes_fast/compressed/latest_net_G.pth \
  --config_str $1 --ngf 64
