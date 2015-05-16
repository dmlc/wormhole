#!/bin/bash
cd `dirname $0`

if [ ! -d `pwd`/../../repo/ps-lite ]; then
    ./build_deps.sh
fi


# set the config.mk
cd ../../
echo "Generate a default config at `pwd`/config.mk"
deps_path=`pwd`/repo/ps-lite/deps
cat <<< "# sample config for learn/linear
CC = gcc
CXX = g++
USE_GLOG = 1
DEPS_PATH = $deps_path
STATIC_DEPS = 1
USE_HDFS = 0
USE_S3 = 0
USE_KEY32 = 1" >config.mk
cat config.mk
cd learn/linear

make -j4
