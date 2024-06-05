#!/usr/bin/env bash
FILE=$1

if [[ $FILE != "cityscapes" && $FILE != "night2day" && $FILE != "edges2handbags" && $FILE != "edges2shoes" && $FILE != "facades" && $FILE != "maps" && $FILE != "edges2handbags-r" && $FILE != "edges2shoes-r" && $FILE != "edges2shoes-v" ]]; then
  echo "Available datasets are cityscapes, night2day, edges2handbags, edges2shoes, edges2handbags-r, edges2shoes-r, edges2shoes-v, facades, maps"
  echo "edges2shoes-r and edges2handbags-r are the redistributed datasets of edges2shoes and edges2handbags which have a large validation set"
  echo "edges2shoes-v only contains the validation set of edges2shoes-r"
  exit 1
fi

if [[ $FILE == "cityscapes" ]]; then
  echo "Due to license issue, we cannot provide the Cityscapes dataset from our repository. Please download the Cityscapes dataset from https://cityscapes-dataset.com, and use the script ./datasets/prepare_cityscapes_dataset.py."
  echo "You need to download gtFine_trainvaltest.zip and leftImg8bit_trainvaltest.zip. For further instruction, please read ./datasets/prepare_cityscapes_dataset.py"
  exit 1
fi

echo "Specified [$FILE]"

if [[ $FILE == "edges2handbags-r" || $FILE == "edges2shoes-r" || $FILE == "edges2shoes-v" ]]; then
  prefix=https://huggingface.co/mit-han-lab/gan-compression/resolve/main/datasets/
else
  prefix=http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/
fi

URL=$prefix$FILE.tar.gz
TAR_FILE=./database/$FILE.tar.gz
TARGET_DIR=./database/$FILE/

mkdir -p ./database
wget -N $URL -O $TAR_FILE
mkdir -p $TARGET_DIR
tar -zxvf $TAR_FILE -C ./database/
rm $TAR_FILE

cd "./database/$FILE" || exit

if [ -e "test" ] && [ ! -e "val" ]; then
  ln -s "test" "val"
fi
