#!/bin/bash
if [ "$#" -lt 2 ];
then
	echo "Usage: <nworkers> <path_in_HDFS> [param=val]"
	exit -1
fi

# put the local training file to HDFS
hadoop fs -rm -r -f $2/mushroom.fm.model

hadoop fs -put ../data/agaricus.txt.train $2/data
hadoop fs -put ../data/agaricus.txt.test $2/data

# submit to hadoop
../../repo/dmlc-core/tracker/dmlc_yarn.py -n $1 --vcores 2 ./fm.dmlc data=hdfs://$2/data/agaricus.txt.train val_data=hdfs://$2/data/agaricus.txt.test model_out=hdfs://$2/mushroom.fm.model max_lbfgs_iter=50 nfactor=8 early_stop=10 "${*:3}"


# get the final model file
hadoop fs -get $2/mushroom.fm.model ./fm.model

../../repo/dmlc-core/yarn/run_hdfs_prog.py ./fm.dmlc data=../data/agaricus.txt.test task=pred model_in=fm.model
../../repo/dmlc-core/yarn/run_hdfs_prog.py ./fm.dmlc task=dump model_in=fm.model name_dump=weight.txt
