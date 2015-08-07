Input Data
==========

Wormhole supports various input data sources and formats.

Data Formats
------------

Both text and binary formats are supported.

LIBSVM
~~~~~~

Wormhole supports a more general version of the LIBSVM format.  Each example is
presented as a text line::

  label feature_id[:weight] feature_id[:weight] ... feature_id[:weight]

label
  a ``float`` label
feature_id
  a ``unsigned 64-bit integer`` feature index. It is not required to be
  continuous.
weight:
  the according ``float`` weight, which is optional

Compressed Row Block (CRB)
~~~~~~~~~~~~~~~~~~~~~~~~~~

This is a compressed binary data format. One can use ``bin/text2crb`` to convert
any supported data format into it.

Customized Format
~~~~~~~~~~~~~~~~~

Adding a customized format requires only two steps.

1. Define a subclass to implement the function ``ParseNext`` of `ParserImpl
   <https://github.com/dmlc/dmlc-core/blob/master/src/data/parser.h>`_. Examples:

   - Parse the text Criteo CTR dataset `criteo_parser <https://github.com/dmlc/wormhole/blob/master/learn/base/criteo_parser.h>`_
   - Parse the binary ``crb`` format: `crb_parser <https://github.com/dmlc/wormhole/blob/master/learn/base/crb_parser.h>`_

2. Then add the this new parser to a reader. For example,
   adding them in the `minibatch reader <https://github.com/dmlc/wormhole/blob/master/learn/base/minibatch_iter.h>`_

Data Sources
------------

Besides standard filesystems, wormhole supports the following distributed
filesystems.

HDFS
~~~~

To support HDFS, compile with the flag ``USE_HDFS=1`` such as ``make
USE_HDFS=1`` or set the flag in ``config.mk``. An example filename of a HDFS
file ::

  hdfs:///user/you/ctr_data/day_0

Amazon S3
~~~~~~~~~

To supports Amazon S3, compile with the flag ``USE_S3=1``. Besides, one needs to
set the environment variables ``AWS_ACCESS_KEY_ID`` and
``AWS_SECRET_ACCESS_KEY`` properly. For example, add the following two lines in
``~/.bashrc`` (replace the strings with your `AWS credentials
<http://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSGettingStartedGuide/AWSCredentials.html>`_)::

  export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
  export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

An example filename of a S3 file ::

  s3://ctr-data/day_0

Microsoft Azure
~~~~~~~~~~~~~~~


To supports Amazon S3, compile with the flag ``USE_AZURE=1``, which needs to
install SDK (TODO: move to make/deps.mk) ::

  sudo apt-get -y install libboost1.54-all-dev libssl-dev cmake libxml++2.6-dev libxml++2.6-doc uuid-dev

  cd deps && mkdir -p lib include

  git clone https://git.codeplex.com/casablanca
  cd casablanca/Release
  mkdir build.release
  cd build.release
  CXX=g++ cmake .. -DCMAKE_BUILD_TYPE=Release
  make -j8
  cp -r ../include/* ../../../include/
  cd ../../..

  git clone https://github.com/Azure/azure-storage-cpp
  cd azure-storage-cpp/Microsoft.WindowsAzure.Storage
  mkdir build.release
  cd build.release
  CASABLANCA_DIR=.././../../casablanca/ CXX=g++ cmake .. -DCMAKE_BUILD_TYPE=Release
  make -j8
  cp Binaries/libazurestorage* ../../../lib
  cp -r ../includes/* ../../../include/
  cd ../../../..

One also need to
set the environment variables properly
(`About Azure storage account <https://azure.microsoft.com/en-us/documentation/articles/storage-create-storage-account/>`_)::

  export AZURE_STORAGE_ACCOUNT=mystorageaccount
  export AZURE_STORAGE_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
