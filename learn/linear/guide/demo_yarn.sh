#!/bin/bash

../../../dmlc-core/tracker/dmlc_yarn.py \
    -n 2 -s 1 --vcores 1 ../build/async_sgd \
    train_data=\\\"hdfs://./data/agaricus.txt.train\\\" \
    val_data=\\\"hdfs://./data/agaricus.txt.test\\\"
