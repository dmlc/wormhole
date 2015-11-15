#!/bin/bash
if [ "$#" -lt 2 ];
then
	echo "Usage: <nworkers> <path_in_HDFS> [param=val]"
	exit -1
fi

# put the local training file to HDFS
hadoop fs -rm -r -f $2/mushroom.linear.model

hadoop fs -put ../data/agaricus.txt.train $2/data

# submit to hadoop
../../repo/dmlc-core/tracker/dmlc_yarn.py --jobname test -q wajue -n $1 --vcores 2 ./fm.dmlc hdfs://$2/data/agaricus.txt.train model_out=hdfs://$2/mushroom.linear.model nfactor=8 "${*:3}"


# get the final model file
hadoop fs -get $2/mushroom.linear.model ./fm.model

../../repo/dmlc-core/yarn/run_hdfs_prog.py ./fm.dmlc ../data/agaricus.txt.test task=pred model_in=fm.model
