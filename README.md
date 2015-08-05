<img src=wormhole.png width=400/>

[![Build Status](https://travis-ci.org/dmlc/wormhole.svg?branch=master)](https://travis-ci.org/dmlc/wormhole)
[![Documentation Status](https://readthedocs.org/projects/wormhole/badge/?version=latest)](http://wormhole.readthedocs.org/en/latest/)
[![GitHub license](https://raw.githubusercontent.com/dmlc/dmlc.github.io/master/img/apache2.0.svg)](./LICENSE)

Portable, scalable and reliable distributed machine learning.

Wormhole is a place where DMLC projects works together to provide
scalable and reliable machine learning toolkits that can run on various platforms

Features
====
* Portable:
  - Supported platforms: local machine, Apache YARN, MPI and Sungrid Engine
* Rich support of Data Source
  - All projects can read data from HDFS, S3 or local filesystem
* Scalable and Reliable

List of Tools
====
* Boosted Trees (GBDT): [XGBoost: eXtreme Gradient Boosting](learn/xgboost)
* Clustering: [kmeans](learn/kmeans)
* Linear method: [Asynchrouns SGD](learn/linear) [L-BFGS](learn/lbfgs-linear)
* Factorization Machine: [DiFacto](learn/difacto)

Build & Run
====

* Requires a C++11 compiler (e.g.~`g++ >=4.8`) and `git`. Install them on Ubuntu
  >= 13.10

```
sudo apt-get update && sudo apt-get install -y build-essential git
```

* Type `make` to build all deps and tools

* All tools can run both in a laptop and in a cluster. For example, train
logisitic regression using 2 workers and one servers in local machine

```
tracker/dmlc_local.py -n 2 -s 1 bin/linear.dmlc learn/linear/guide/demo.conf
```

* [More tutorials and documents](http://wormhole.readthedocs.org/en/latest/index.html)

Support
====

If you are having issues, please let us [know](https://github.com/dmlc/wormhole/issues).


Contribute
====
- We are actively building new tools. The source codes of all tools are available under [learn/](learn).
- Wormhole depends on other DMLC projects, which are also under active developing
  - [dmlc-core](https://github.com/dmlc/dmlc-core) provides I/O modules and job
    trackers
  - [rabit](https://github.com/dmlc/rabit) provides reliable BSP Allreduce communication.
  - [ps-lite](https://github.com/dmlc/ps-lite) provides the asynchronous key-value
    push and pull for the parameter server framework.
