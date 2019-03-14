#!/usr/bin/env bash

set -e

#sh dataset/download_and_convert_ctscans.sh

DOWN_FILENAME="deeplabv3_pascal_trainval_2018_01_04.tar.gz"
FILENAME="deeplabv3_pascal_trainval"
IMAGENET="imagenet_model"
IMAGENET_TAR=${IMAGNET}".tar.gz"

echo "Downloading models"
wget http://download.tensorflow.org/models/deeplabv3_pascal_trainval_2018_01_04.tar.gz
wget -O ${IMAGENET}.tar.gz http://download.tensorflow.org/models/deeplabv3_xception_2018_01_04.tar.gz

echo "Uncompressing model tar files"

tar -xf ${DOWN_FILENAME}
mkdir -p models
mv ${FILENAME} models/pascal
rm ${DOWN_FILENAME}

tar -xf ${IMAGENET_TAR}
mv xception models/imagenet
rm ${IMAGENET_TAR}
#mv ${IMAGENET}

