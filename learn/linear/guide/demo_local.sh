#!/bin/bash

../../../dmlc-core/tracker/dmlc_local.py \
    -n 2 -s 1 ../build/async_sgd \
    train_data=\\\"../../data/agaricus.txt.train\\\" \
    val_data=\\\"../../data/agaricus.txt.test\\\"
