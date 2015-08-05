Binary Classification on the Criteo CTR Dataset
===============================================

This tutorial gives a step-by-step example for training a binary classifier on
the `Criteo Kaggle CTR competetion dataset
<https://www.kaggle.com/c/criteo-display-ad-challenge/>`_. In this dataset, each
example (text line) presents a displayed ad with the label clicked (+1) or not
(-1). The goal is to predict the probability of being clicked for a new ad.
This is a standard click-through rate (CTR) estimation problem.

In the following we assume a recent Ubuntu (>= 13.10) and bash is used, it should
apply to other Linux distributions and Mac OS X too.

Preparation
-----------

We first build wormhole using 4 threads::

  git clone https://github.com/dmlc/wormhole
  cd wormhole && make deps -j4 && make -j4

Then download the dataset, which has two text files ``train.txt`` and
``test.txt``. Even though wormhole can directly read these two files, we split
``train.txt`` to multiple files to easy training and validation. The following
command divides ``train.txt`` into multiple 300MB size files, and store them in
a compressed row block (crb) format::

  wget https://s3-eu-west-1.amazonaws.com/criteo-labs/dac.tar.gz
  tar -zxvf dac.tar.gz
  mkdir data
  wormhole/bin/text2crb train.txt data/train criteo 300


Linear Method
-------------

We first learn a linear logistic regression using
``linear.dmlc``. We train on the first 20 parts and validate the model on the
last 6 parts. A sparse regularizer :math:`4 |w|_1` is used to control the model
complexity. Furthermore, we solve the problem via asynchronous SGD with
minibatch size 10000 and learning rate 0.1.

Now generate the configuration file (`learn more <../learn/linear.html>`_)::

  cat >train.conf <<EOF
  train_data = "data/train-part_[0-1].*"
  val_data = "data/train-part_2.*"
  data_format = "crb"
  model_out = "model/criteo"
  lambda_l1 = 4
  lr_eta = .1
  minibatch = 10000
  max_data_pass = 1
  EOF


We train the model using 10 workers and 10 servers::

  mkdir model
  wormhole/tracker/dmlc_local.py -n 10 -s 10 wormhole/bin/linear.dmlc train.conf

A possible training log is ::

  2015-07-22 04:50:55,285 INFO start listen on 192.168.0.112:9091
  connected 10 servers and 10 workers
  training #iter = 1
  sec #example delta #ex    |w|_0      logloss     AUC    accuracy
    1  1.8e+06  1.8e+06        30509  0.507269  0.758684  0.769462
    2  3.7e+06  1.9e+06        50692  0.469855  0.782046  0.780102
    3  5.5e+06  1.9e+06        70856  0.462922  0.785047  0.784311
    4  7.5e+06    2e+06        85960  0.462718  0.786288  0.783614
    ...
   18  3.4e+07    2e+06       231968  0.453590  0.793880  0.789032
   19  3.6e+07    2e+06       242017  0.454674  0.794033  0.788652
   20  3.7e+07  8.4e+05       248066  0.461133  0.791255  0.784265
  validating #iter = 1
  sec #example delta #ex    |w|_0      logloss     AUC    accuracy
   30  4.6e+07  9.3e+06       248066  0.459048  0.791334  0.785863
  hit max number of data passes
  saving final model to model/criteo
  training is done!

Then we can perform prediction using the trained model. Generate the prediction
config file ::

  cat >pred.conf <<EOF
  val_data = "test.txt"
  data_format = "criteo_test"
  model_in = "model/criteo"
  pred_out = "output/criteo"
  EOF

and predict::

  mkdir output
  wormhole/tracker/dmlc_local.py -n 10 -s 10 wormhole/bin/linear.dmlc pred.conf
  cat output/criteo* >pred.txt


Then the i-th line  of ``pred.txt`` will contains the prediction :math:`p=\langle
w, x \rangle` for be i-th example (line) in ``test.txt``. We can convert it into
a probability by :math:`1/(1+\exp(-p))`.

Factorization Machine
---------------------

Factorization machine learns an additional embedding comparing to the linear
model, which catches the high-order interactions between features. The usage of
``difacto.dmlc`` is similar to ``linear.dmlc``. First generate the configure
file ::

  cat >train.conf <<EOF
  train_data = "data/train-part_[0-1].*"
  val_data = "data/train-part_2.*"
  data_format = "crb"
  model_out = "model/criteo"
  embedding {
    dim = 16
    threshold = 16
    lambda_l2 = 0.0001
  }
  lambda_l1 = 4
  lr_eta = .01
  max_data_pass = 1
  minibatch = 1000
  early_stop = 1
  EOF

Then train the model::

  wormhole/tracker/dmlc_local.py -n 10 -s 10 wormhole/bin/difacto.dmlc train.conf

We can reuse the previous ``pred.conf`` for prediction::

  wormhole/tracker/dmlc_local.py -n 10 -s 10 wormhole/bin/difacto.dmlc pred.conf
  cat output/criteo* >pred.txt

What's Next?
------------

- `Use another dataset with different formats or storing on HDFS, Amazon S3 <../common/input.html>`_

- `Train the model over multiple machines on Apache Yarn, Amazon EC2 <../common/build.html#run>`_
