#!/bin/bash

../../../dmlc-core/tracker/dmlc_yarn.py \
    -n 2 -s 1 --vcores 1 ../build/async_sgd demo.conf
