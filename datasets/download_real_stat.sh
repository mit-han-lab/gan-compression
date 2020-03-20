#!/usr/bin/env bash

echo "The real state files are in the format dataset_side.npz. Please specify the dataset and the side."
echo "Supported file: cityscapes-A.npz edges2shoes_B.npz edges2shoes-r_B.npz edges2handbags_B.npz edges2handbags-r_B.npz"
echo "                facades_A.npz horse2zebra_A.npz horse2zebra_B.npz maps_A.npz maps_B.npz"

FILE=$1
SIDE=$2

prefix=https://hanlab.mit.edu/files/gan_compression/real_stat/

URL=${prefix}${FILE}_${SIDE}.npz
TARGET_DIR=./real_stat/
TARGET_FILE=./real_stat/${FILE}_${SIDE}.npz
mkdir -p $TARGET_DIR
wget -N $URL -O $TARGET_FILE
