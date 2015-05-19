# Configuration

The configuration is specified by the protobuf file
[config.proto](../proto/config.proto). It contains three parts.

## Input and output

This part specifies both input data and output model. They can be on
local disk, HDFS (starts with `hdfs://`) or Amazon S3 (starts with `s3://`).
Regular expression is supported to match files. such as `.*` or `[1-5]` (if
using `gcc` the latter needs version >= 4.9).

```bash
train_data = "s3://ctr-data/criteo/day_.*"
val_data = "s3://ctr-data/criteo-val/day_.*"
data_format = "criteo"
num_parts_per_file = 50
model_out = "s3://ctr-data/model/online"
```

- `data_format` supports `libsvm`, `criteo` and `criteo_rec` (recordio
  format).
- `num_parts_per_file` when the number of files is smaller than the number of
  workers, use it to virtually split a file into several parts.
- `model_out` each server will write its model part into
  `${model_out}_${server_id}`. In default it uses text format, each line has a
  `feature_id weight` pair.

See [dev.md](dev.md) on how to add more data formats.

## Objective function

A objective function has two parts: loss and penalty.

```bash
loss = LOGIT
penalty = L1
lambda = 1
lambda = .1
```

- `loss` currently supports logistic loss `LOGIT`, square hinge loss
  `SQUARE_HINGE`.
- `penalty` currently supports `L1` penalty. On the examples, the penalty equals
  to `1 * |w|_1 + .1 * 1/2||w||_2^2`.

See [dev.md](dev.md) on how to add more loss and penalty functions.

## Optimization Methods

We implemented asynchronous minibatch SGD. More are coming soon.

```bash
algo = FTRL
max_data_pass = 1
minibatch = 100000
lr_eta = .1
lr_beta = 1
disp_itv = 1
max_delay = 4
```

- `algo`. `FTRL` is a variant of the adaptive SGD which produces
  good sparse solution.
- `max_data_pass` the maximal number of data passes. When the data is large and
  the online solver is used, one data pass is often good enough.
- `minibatch` the minibatch size, which is a trade-off between algorithm
  convergence (prefer small values) and system performance (prefer large
  values). Usually a number between `10K` and `100K` is a good choice.
- `max_delay` the max number of parallel minibatches a worker can
  process. Again, it is a convergence and performance trade-off
- learning rate `lr_eta` and `lr_beta`. In the online solver, the learning rate is `lr_eta /
  (lr_beta + x)` where `x` depends on the progress, such as `x=sqrt(t)` or
  `x=||grad||`. Often `lr_beta=1` is a good choice, but we may need to tune
  `lr_eta` for faster convergence.
- `disp_itv` prints the progress in every `disp_itr` seconds.
