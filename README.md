![Wormhole](wormhole.png)

[![Build Status](https://travis-ci.org/dmlc/xgboost.svg?branch=master)](https://travis-ci.org/dmlc/xgboost)
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
* Requires a C++11 compiler such as `g++ >=4.8` and `git`
  - On Ubuntu >= 13.10
  ```
  sudo apt-get update && sudo apt-get install -y build-essential git
  ```
  - On older Ubuntus
  ```
  sudo apt-get install python-software-properties
  sudo add-apt-repository ppa:ubuntu-toolchain-r/test
  sudo apt-get update && sudo apt-get -y install gcc-4.8 git make
  sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 50
  ```
* Type `make` or `make xgboost` to selectly build one tool
* More options are available
  - `make -j4` uses 4 threads
  - `make CXX=gcc-4.9` changes the compiler
  - `make DEPS_PATH=your_path` changes the path of the deps libaries
  - `make USE_HDFS=1` to enable read/write HDFS. Make sure hadoop is installed.
  - `make USE_S3=1` to enable read/write AWS S3 files. You may need to install
    `libcurl4-openssl-dev` first via `sudo apt-get install libcurl4-openssl-dev`

How to Submit Jobs
====
* make sure `repo/dmlc-core` exist
  - type ```make repo/dmlc-core``` to get it
* Use the submission script in ```tracker/``` to submit job to the platform of your choice

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
  - [parameter-server](https://github.com/dmlc/parameter-server) provides asynchronize parameter server abstraction.
