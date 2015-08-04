Build & Run
================

Prerequisites
-------------------

Wormhole can be built on both Linux and Mac OS X. Some apps are also tested on Windows. To build wormhole, both ``git`` and a recent C++ compiler supporting ``C++11``,
such as ``g++ >= 4.8`` and ``clang >= 3.5``, are required. Install them on

1. Ubuntu >= 13.10::

     $ sudo apt-get update && sudo apt-get install -y build-essential git

2. Older version Ubuntu via `ppa:ubuntu-toolchain-r/test <http://ubuntuhandbook.org/index.php/2013/08/install-gcc-4-8-via-ppa-in-ubuntu-12-04-13-04/>`_:
3. Centos via `devtoolset <http://linux.web.cern.ch/linux/devtoolset/>`_
4. Mac OS X: can either use the ``clang`` provided by command line tools or download
   a compiled gcc from `hpc.sourceforge.net <http://hpc.sourceforge.net/>`_

Build
------

Type ``make`` to build all apps. It may take several minutes for the first
time due to building all dependencies such as ``gflags``. There are several options for advanced usages.

make xgboost
  selectively builds xgboost. Similarly for linear, difactor, ...

make -j4
  uses 4 threads for parallel building. For the first building, we suggest to build
  deps and apps separately: ``make deps -j4 && make -j4``

make CXX=g++-4.9
  uses a different compiler

make DEPS_PATH=your_path
  changes the path of the deps. In default all deps will be installed on
  ``wormhole/deps``. We can change the path if them are installed on another
  place.

make USE_HDFS=1
  supports read/write HDFS. It requires ``libhdfs``, which is often installed
  with Hadoop.

make USE_S3=1
  supports read/write AWS S3. ``libcurl4-openssl-dev`` is required, it can be
  installed via ``sudo apt-get install libcurl4-openssl-dev`` on Ubuntu

Run
---

Wormhole runs both in a laptop and in a cluster. A typical command to run a
application::

  $ tracker/dmlc_xxx.py -n num_workers [-s num_servers] app_bin app_conf

`tracker/dmlc_xxx.py`
  the tracker provided by dmlc-core to launch jobs on various platforms
`-n`
  number of workers
`-s`
  number of servers. Only required for parameter server applications
`app_bin`
  the binary of the application, which is available under ``bin/``
`app_conf`
  the text configuration file specifying dataset and learning method, see
  each app's documents for details

Local machine
~~~~~~~~~~~~~

The following command runs linear logistic regression using two workers and a
single server on a small dataset::

  $ tracker/dmlc_local.py -n 2 -s 1 bin/linear.dmlc learn/linear/guide/demo.conf

Apache Yarn
~~~~~~~~~~~

First make sure the environments ``HADOOP_HOME`` and ``JAVA_HOME`` are set
properly. Next compile the Yarn tracker::

  $ cd repo/dmlc-core/yarn && ./build.sh

Then a Yarn job can be submitted via ``tracker/dmcl_yarn.py``. For
example, the following codes run xgboost on Yarn

.. code-block:: bash

   hdfs_path=/your/path

   hadoop fs -mkdir ${hdfs_path}/data
   hadoop fs -put learn/data/agaricus.txt.train ${hdfs_path}/data
   hadoop fs -put learn/data/agaricus.txt.test ${hdfs_path}/data

   tracker/dmlc_yarn.py  -n 4 --vcores 2 bin/xgboost.dmlc \
     learn/xgboost/mushroom.hadoop.conf nthread=2 \
     data=hdfs://${hdfs_path}/data/agaricus.txt.train \
     eval[test]=hdfs://${hdfs_path}/data/agaricus.txt.test \
     model_out=hdfs://${hdfs_path}/mushroom.final.model

Run ``tracker/dmlc_yarn.py -h`` for more details.

Sun Grid Engine
~~~~~~~~~~~~~~~

Use ``tracker/dmlc_sge.py``

MPI
~~~

Wormhole can be run over multiple machines via ``mpirun``, which is often
convenient for a small cluster. Assume file ``hosts`` stores the hostnames of
all machines, then use::

   $ tracker/dmlc_mpi.py -n num_workers -s num_servers -H hosts bin conf

to launch wormhole on these machines. See next section for an example to setup a
cluster with mpirun.

Setup an EC2 Cluster from Scratch
---------------------------------

In this section we give a tutorial to setup a small cluster and launch wormhole
jobs on Amazon EC2.

1. Assume all data are stored Amazon S3.
2. Use a middle range instance as the master node to build wormhole and submit jobs,
   and several high end instances to do the computations.
3. Use ``NFS`` to dispatch binaries and configurations and ``mpirun`` to launch
   jobs.


Setup the master node
~~~~~~~~~~~~~~~~~~~~~

First launch an Ubuntu 14.04 instance as the master node. It is mainly used for
compiling codes, a middle end instance such as `c4.xlarge` is often good
enough. Install required libraries via::

  $ sudo apt-get update && sudo apt-get install -y build-essential git libcurl4-openssl-dev

Then build wormhole with S3 support::

  $ git clone https://github.com/dmlc/wormhole.git
  $ cd wormhole && make deps -j4 && make -j4 USE_S3=1

Next setup NFS::

  $ sudo apt-get install nfs-kernel-server mpich2
  $ echo "/home/ubuntu/  *(rw,sync,no_subtree_check)" | sudo tee /etc/exports
  $ sudo service nfs-kernel-server start

Finally copy the `pem` file used to access the master node to master node's
`~/.ssh/id_rsa` so that this node can access to all other machines.

Setup the slave nodes
~~~~~~~~~~~~~~~~~~~~~

First launch several Ubuntu 12.04 instances with the same pem file as the slaves
nodes. High-end instances such as c4.4xlarge and c4.8xlarge are
recommended. Save their private IPs in file `hosts`::

  $ cat hosts
  172.30.0.172
  172.30.0.171
  172.30.0.170

Then install both NFS and mpirun on these slave nodes. Assume the master node has
private IP ``172.30.0.160``::

  while read h; do
    echo $h
    ssh -o StrictHostKeyChecking=no $h <<'ENDSSH'
  sudo apt-get update
  sudo apt-get install -y nfs-common mpich2
  sudo mount 172.30.0.160:/home/ubuntu /home/ubuntu
  ENDSSH
  done <hosts

Next install depended libraries on all slave nodes::

  $ mpirun -hostfile hosts sudo apt-get install -y build-essential libcurl4-openssl-dev

Put all things together
~~~~~~~~~~~~~~~~~~~~~~~

Test if everything is OK::

  $ mpirun -hostfile hosts uname -a
  $ mpirun -hostfile hosts ldd wormhole/bin/linear.dmlc

Now we can submit jobs from the master node via::

  $ wormhole/tracker/dmlc_mpi.py -n ? -s ? -H hosts wormhole/bin/? ?.conf
