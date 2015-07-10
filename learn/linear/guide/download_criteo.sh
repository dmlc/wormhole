#!/bin/bash

dir=`dirname "$0"`
mkdir -p $dir/criteo && cd $dir/criteo

if [ ! -f train.txt ]; then
    if [ ! -f dac.tar.gz ]; then
        wget https://s3-eu-west-1.amazonaws.com/criteo-labs/dac.tar.gz
    fi
    tar -zxvf dac.tar.gz
fi

echo "split train.txt..."
mkdir -p train
split -n l/18 --numeric-suffixes=1 --suffix-length=3 train.txt train/part-

echo "make a test set"
mkdir -p test
mv train/part-01[7-8] test
