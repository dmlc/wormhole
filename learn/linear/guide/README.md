# Tutorial to run linear method

In this tutorial we show how to solve the following linear method

```tex
min_w sum_i loss(y_i, <w, x_i>) + penalty(w)
```

where `(x_i, y_i)` is the data pair and vector `w` is the model we will to
learn.

We will use the Kaggle
[Criteo CTR](https://www.kaggle.com/c/criteo-display-ad-challenge) dataset as an
example here, which has around 46 millions exmaples and millions of
features. There is a script to download this dataset
[download_criteo.sh](./download_criteo.sh).

## Configuration

There is an example configuration [criteo.conf](criteo.conf) which trains a
sparse logistic regression using online method.

It has three parts

```
# Data
train_data = "../guide/criteo/train/part-.*"
val_data = "../guide/criteo/test/part-.*"
data_format = "criteo"
num_parts_per_file = 4

# Model
loss = LOGIT
penalty = L1
lambda = 1
lambda = .1

# Solver
algo = FTRL
max_data_pass = 1
minibatch = 10000
lr_eta = .1
lr_beta = 1
show_prog = 1
max_delay = 1
```

- Data. The data can be in local filesystem, hdfs (start with `hdfs://`), s3 (start
  with `s3://`). It support regular expression to match the files.

  The other data format can be `libsvm`, `csv`, ...

- Model.  Here we use logistic loss and the penalty `1*|w|_1 + .1 * 1/2||w||^2_2`.

- Solver. We use an online solver

## Run

We have a single machine implementation [ftrl.cc](../sgd/ftrl.cc) which helps to
read the code. To run the algorithm, simply run
```
../build/ftrl -conf criteo.conf
```

We can use the
[dmlc tracker](https://github.com/dmlc/dmlc-core/tree/master/tracker) to start
the distributed version `async_sgd`. Or simply start 2 servers and 2 workers in the local
machine

```
./local.sh 2 2 ../build/async_sgd -conf criteo.conf
```

(PS. using 1 worker and `max_delay=0`, `async_sgd` is identical to `ftrl`).


## Related links

- [Criteo](http://labs.criteo.com/downloads/download-terabyte-click-logs/) released a bigger dataset with billions of examples, which is better for
testing the performance.
- Use a block coordinate descent solver
  [link](https://github.com/dmlc/parameter_server/tree/master/example/linear/criteo)
- Use a L-BFGS solver [link](https://github.com/dmlc/wormhole/tree/master/learn/lbfgs-linear)
