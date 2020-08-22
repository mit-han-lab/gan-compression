#!/usr/bin/env bash
set -ex
conda install pytorch==1.4.0 torchvision==0.5.0 -c pytorch
conda install tqdm scipy tensorboard
conda install tensorboardx -c conda-forge
pip install opencv-python dominate wget pycocotools scikit-image torchprofile
