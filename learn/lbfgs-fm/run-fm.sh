#!/bin/bash
if [[ $# -lt 1 ]]
then
    echo "Usage: nprocess"
    exit -1
fi

rm -rf *.model
k=$1

# run fm model, the program will automatically split the inputs
../../repo/dmlc-core/tracker/dmlc_local.py -n $k fm.dmlc data=../data/agaricus.txt.train val_data=../data/agaricus.txt.test reg_L1=1 early_stop=2 

echo "train done"
./fm.dmlc data=../data/agaricus.txt.test task=pred model_in=final.model
