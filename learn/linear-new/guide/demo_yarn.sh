#!/bin/bash
if [ "$#" -lt 1 ]; then
	echo "Usage: <path_in_HDFS>"
	exit -1
fi

hdfs_dir=$1

${HADOOP_HOME}/bin/hadoop fs -rm -r -f $hdfs_dir/data
${HADOOP_HOME}/bin/hadoop fs -mkdir $hdfs_dir/data
${HADOOP_HOME}/bin/hadoop fs -put ../../data/agaricus.txt.train $hdfs_dir/data
${HADOOP_HOME}/bin/hadoop fs -put ../../data/agaricus.txt.test $hdfs_dir/data

cat <<< "
train_data = \"hdfs://${hdfs_dir}/data/agaricus.txt.train\"
val_data = \"hdfs://${hdfs_dir}/data/agaricus.txt.test\"
max_data_pass = 3
" >demo_hdfs.conf

../../../dmlc-core/tracker/dmlc_yarn.py --vcores 1 \
    -n 1 -s 1 ../build/async_sgd demo_hdfs.conf
