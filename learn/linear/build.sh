#!/bin/bash
cd `dirname $0`

echo "Clone dmlc-core, ps-lite and build the deps"
make -C ../.. dmlc-core
make -C ../.. repo/ps-lite
../../repo/ps-lite/make/install_deps.sh

# set the config.mk
cd ../../
echo "Generate a default config at `pwd`/config.mk"
deps_path=`pwd`/repo/ps-lite/deps
cat <<< "# sample config for learn/linear
CC = gcc
CXX = g++
USE_GLOG = 1
DEPS_PATH = $deps_path
USE_HDFS = 0
USE_S3 = 0
USE_KEY32 = 1" >config.mk
cat config.mk
cd learn/linear

make -j4
