#!/usr/bin/env bash
python export.py \
  --input_path supernet-2000000/checkpoints/latest_net_G.pth \
  --output_path pretrained/munit/edges2shoes-r/compressed/latest_net_G.pth \
  --ngf 64 --model mobile_munit --config_str $1
