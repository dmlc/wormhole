<img src=wormhole.png width=400/>

[![Build Status](https://travis-ci.org/dmlc/wormhole.svg?branch=master)](https://travis-ci.org/dmlc/wormhole)
[![Documentation Status](https://readthedocs.org/projects/wormhole/badge/?version=latest)](https://readthedocs.org/projects/wormhole/?badge=latest)

Portable, scalable and reliable distributed machine learning.

Wormhole is a place where DMLC projects works together to provide
scalable and reliable machine learning toolkits that can run on various platforms

Features
====
* Portable:
  - Supported platforms: YARN, MPI and Sungrid Engine
  - Planned: docker support
* Rich support of Data Source
  - All projects can read data from HDFS, S3 or local filesystem
* Scalable and Reliable

List of Tools
====
* Boosted Trees (GBDT): [XGBoost: eXtreme Gradient Boosting](learn/xgboost)
* Clustering: [kmeans](learn/kmeans)
* Linear method: [Asynchrouns SGD](learn/linear) [L-BFGS](learn/lbfgs-linear)
* Factorization Machine: [DiFacto](learn/difacto)

Build
====
* Requires a C++11 compiler such as `g++ >=4.8` and `git`. You can install them via
```
sudo apt-get update && sudo apt-get install -y build-essential git
```
on Ubuntu >= 13.10. For
[older Ubuntu](http://ubuntuhandbook.org/index.php/2013/08/install-gcc-4-8-via-ppa-in-ubuntu-12-04-13-04/),
[Centos](http://linux.web.cern.ch/linux/devtoolset/),
[Mac Os X](http://hpc.sourceforge.net/).

* Type `make` to build all deps and tools, it takes several minutes in the fist time.

* More options are available
  - `make xgboost` to selectly build one tool
  - `make -j4` uses 4 threads
  - `make CXX=gcc-4.9` changes the compiler
  - `make DEPS_PATH=your_path` changes the path of the deps libaries
  - `make USE_HDFS=1` to enable read/write HDFS. Make sure hadoop is installed.
  - `make USE_S3=1` to enable read/write AWS S3 files. You may need to install
    `libcurl4-openssl-dev` first via `sudo apt-get install libcurl4-openssl-dev`

How to Submit Jobs
====
* All tools can run both in your laptop and in a cluster. For example, the following
command runs linear logistic regression using single worker and server on a
small dataset.
```
tracker/dmlc_local.py -n 1 -s 1 bin/linear.dmlc learn/linear/guide/demo.conf
```

* See more trackers in ```tracker/```

Contributing
====
* We believe that we can create machine learning tools that are portable and works with each other.
* Contributing of machine learning projects, tutorials and to core dmlc projects are welcomed.
  - All machine learning projects can depends on dmlc-core, rabit or parameter-server

Project Structure
====
* [learn](learn) contains simple but powerful learning tools in wormhole
* [repo](repo) is used to clone other DMLC repos that wormhole can depend on
* Depending DMLC Libraries
  - [dmlc-core](https://github.com/dmlc/dmlc-core) gives the core modules of most DMLC projects.
  - [rabit](https://github.com/dmlc/rabit) provides reliable BSP Allreduce communication.
  - [parameter-server](https://github.com/dmlc/parameter_server) provides asynchronize parameter server abstraction.
