#!/bin/bash
if [[ $# -lt 1 ]]
then
    echo "Usage: nprocess"
    exit -1
fi

rm -rf *.model
k=$1

# run fm model, the program will automatically split the inputs
../../repo/dmlc-core/tracker/dmlc_local.py -n $k fm.dmlc ../data/agaricus.txt.train reg_L1=1 

echo "train done"
./fm.dmlc ../data/agaricus.txt.test task=pred model_in=final.model
