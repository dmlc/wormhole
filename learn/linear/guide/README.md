# Tutorial to run linear method

In this tutorial we show how to solve the following linear method

```tex
min_w sum_i loss(y_i, <w, x_i>) + penalty(w)
```

where `(x_i, y_i)` is the data pair and vector `w` is the model we will to
learn.

## Build

Use [build.sh](../build.sh) to build this applications. It will first download
[dmlc-core](https://github.com/dmlc/dmlc-core) and
[ps-lite](https://github.com/dmlc/ps-lite), next generate a default build configuration,
and then build all.

In the default configuration, it disables HDFS (`USE_HDFS=0`) and S3
(`USE_S3=0`). And since criteo only contains millions of unique features, it uses
32bit integers for feature ID to save network bandwidth
(`USE_KEY32=1`). (Parameter server use 64bit feature ID in default.)


## Run

[demo_local.sh](demo_local.sh) will train the sample dataset in local, and
[demo_yarn.sh](demo_yarn.sh) will run the job on yarn. The latter needs to set
`USE_HDFS=1` during compling. (If you are using CDH hadoop, then you need put
`hdfs.h` in `dmlc-core/include`, see this
[issue](https://github.com/dmlc/dmlc-core/issues/10) for more details.)

## Train on Criteo CTR dataset

Next will use the Kaggle
[Criteo CTR](https://www.kaggle.com/c/criteo-display-ad-challenge) dataset as an
example, which has around 46 millions exmaples and millions of features. There
is a script to download this dataset [download_criteo.sh](./download_criteo.sh).

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
disp_itv = 1
max_delay = 1
```

- Data. The data can be in local filesystem, hdfs (start with `hdfs://`), s3 (start
  with `s3://`). It support regular expression to match the files.

  The other data format can be `libsvm`, `csv`, ...

- Model.  Here we use logistic loss and the penalty `1*|w|_1 + .1 * 1/2||w||^2_2`.

- Solver. We use an online solver

To start the job on local using 2 servers and 2 workers.
```
../../../dmlc-core/tracker/dmlc_local.py \
    -n 2 -s 2 ../build/async_sgd \
    -conf criteo.conf
```


There are various
[trackers](https://github.com/dmlc/dmlc-core/tree/master/tracker) to start
the jobs on Yarn, MPI, SUN SGE, ...

PS: there is single machine implementation [ftrl.cc](../sgd/ftrl.cc) which might
help read and debug the codes. Run it via
```
../build/ftrl -conf criteo.conf
```
It is identical to using 1 worker and set `max_delay=0` on `async_sgd`.


## Related links

- [Criteo](http://labs.criteo.com/downloads/download-terabyte-click-logs/) released a bigger dataset with billions of examples, which is better for
testing the performance.
- Use a block coordinate descent solver
  [link](https://github.com/dmlc/parameter_server/tree/master/example/linear/criteo)
- Use a L-BFGS solver [link](https://github.com/dmlc/wormhole/tree/master/learn/lbfgs-linear)
