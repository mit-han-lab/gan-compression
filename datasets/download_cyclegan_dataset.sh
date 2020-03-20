#!/usr/bin/env bash
FILE=$1

if [[ $FILE != "ae_photos" && $FILE != "apple2orange" && $FILE != "summer2winter_yosemite" && $FILE != "horse2zebra" && $FILE != "monet2photo" && $FILE != "cezanne2photo" && $FILE != "ukiyoe2photo" && $FILE != "vangogh2photo" && $FILE != "maps" && $FILE != "cityscapes" && $FILE != "facades" && $FILE != "iphone2dslr_flower" && $FILE != "mini" && $FILE != "mini_pix2pix" && $FILE != "mini_colorization" ]]; then
  echo "Available datasets are: apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, ae_photos"
  exit 1
fi

echo "Specified [$FILE]"
URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/$FILE.zip
ZIP_FILE=./database/$FILE.zip
TARGET_DIR=./database/$FILE/

mkdir -p ./database
wget -N $URL -O $ZIP_FILE
mkdir -p $TARGET_DIR
unzip $ZIP_FILE -d ./database/
rm $ZIP_FILE

cd "./database/$FILE" || exit

if [ -e "testA" ] && [ ! -e "valA" ]; then
  ln -s "testA" "valA"
fi

if [ -e "testB" ] && [ ! -e "valB" ]; then
  ln -s "testB" "valB"
fi