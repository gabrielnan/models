#!/usr/bin/env bash

set -e

#sh dataset/download_and_convert_ctscans.sh

DOWN_FILENAME="deeplabv3_pascal_trainval_2018_01_04.tar.gz"
FILENAME="deeplabv3_pascal_trainval"

echo "Downloading model"
wget http://download.tensorflow.org/models/deeplabv3_pascal_trainval_2018_01_04.tar.gz

echo "Uncompressing model tar file"

tar -xf ${DOWN_FILENAME}
mkdir -p models
mv ${FILENAME} models/pascal
rm ${DOWN_FILENAME}

