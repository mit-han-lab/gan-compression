#!/usr/bin/env bash
python export.py \
--input_path logs/pix2pix/edges2shoes-r/finetune/checkpoints/latest_net_G.pth \
--output_path logs/pix2pix/edges2shoes-r/compressed/compressed_net_G.pth \
--config_str $1