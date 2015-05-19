# Tutorial to run linear method

In this tutorial we show how to solve the following linear method

```tex
min_w sum_i loss(y_i, <w, x_i>) + penalty(w)
```

where `(x_i, y_i)` is a data pair and vector `w` is the model we will
learn.

## Build & Run

Use [build.sh](../build.sh) to build, and
[demo_local.sh](demo_local.sh) to train on the sample dataset in local machine, or
[demo_yarn.sh](demo_yarn.sh) to run the job on Yarn.

## More

- [A Step-by-step tutorial](./criteo.md) of training sparse logistic regression using the Criteo
  Terabytes CTR dataset on Amazon EC2. It processes 10 millions examples per
  second using only 5 machines.
- [More about build](./build.md)
- [Job configuration](./conf.md)
- [How to contribute](./dev.md)
- Use a [block coordinate descent solver](https://github.com/dmlc/parameter_server/tree/master/example/linear/criteo)
- Use a [L-BFGS solver](https://github.com/dmlc/wormhole/tree/master/learn/lbfgs-linear)
