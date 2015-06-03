# Build

We needs a CXX11 C compiler to build the codes, which is contained on recent
Linux. For example, on `Ubuntu >= 13.10`, simply

```bash
sudo apt-get update && sudo apt-get install -y build-essential git
```

For old Linux systems, we can either download the package
e.g. [Centos](http://linux.web.cern.ch/linux/devtoolset/),
[Ubuntu](http://ubuntuhandbook.org/index.php/2013/08/install-gcc-4-8-via-ppa-in-ubuntu-12-04-13-04/),
[Mac Os X](http://hpc.sourceforge.net/), or building from source, such as for
[Centos](http://www.codersvoice.com/a/webbase/install/08/202014/131.html).


The default [build.sh](../build.sh) will first use [build_deps.sh](../build.sh)
to download and build [dmlc-core](https://github.com/dmlc/dmlc-core) and
[ps-lite](https://github.com/dmlc/ps-lite).

It then generates a config file to build the system.
```bash
deps_path=`pwd`/repo/ps-lite/deps
CC = gcc
CXX = g++
USE_GLOG = 1
DEPS_PATH = $deps_path
STATIC_DEPS = 1
USE_HDFS = 0
USE_S3 = 0
USE_KEY32 = 1
```

We can add HDFS support by changing to `USE_HDFS = 1` when `libhdfs` is
available. `libhdfs` is often shipped with Hadoop. However, for CDH Hadoop, we
need put `hdfs.h` in `dmlc-core/include`, see this
[issue](https://github.com/dmlc/dmlc-core/issues/10) for more details.

Similarly we can turn on `S3` support. It relies on
`libcurl4-openssl-dev`.


Here we also 32bit integers for feature ID to save network bandwidth via
`USE_KEY32 = 1` (In default we use 64bit integers). Remove it if the data set
contains billions of features.
