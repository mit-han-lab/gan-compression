#!/usr/bin/env bash
python distill.py --dataroot database/edges2shoes-r \
--distiller resnet \
--log_dir logs/pix2pix/edges2shoes-r/distill \
--batch_size 4 \
--restore_teacher_G_path logs/pix2pix/edges2shoes-r/mobile/checkpoints/latest_net_G.pth \
--restore_pretrained_G_path logs/pix2pix/edges2shoes-r/mobile/checkpoints/latest_net_G.pth \
--restore_D_path logs/pix2pix/edges2shoes-r/mobile/checkpoints/latest_net_D.pth \
--real_stat_path real_stat/edges2shoes-r_B.npz