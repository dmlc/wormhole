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

## Related links

- [Step-by-step tutorial](./criteo.md) on training sparse logistic regression on the Criteo
  Terabytes CTR dataset on EC2.
- Use a block coordinate descent solver
  [link](https://github.com/dmlc/parameter_server/tree/master/example/linear/criteo)
- Use a L-BFGS solver [link](https://github.com/dmlc/wormhole/tree/master/learn/lbfgs-linear)
