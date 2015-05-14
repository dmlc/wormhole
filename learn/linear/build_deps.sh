#!/bin/bash
cd `dirname $0`
echo "Clone dmlc-core, ps-lite and build the deps"
make -C ../.. dmlc-core
make -C ../.. repo/ps-lite
../../repo/ps-lite/make/install_deps.sh
