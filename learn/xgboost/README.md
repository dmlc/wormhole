Distributed XGBoost
====
This folder contains information of Distributed XGBoost (Distributed GBDT).

[XGBoost](https://github.com/dmlc/xgboost) is a project that provides heavily optimized gradient boosting tools.
* The single machine version is available in python, R and Julia.
* XGBoost features at extremely fast speed and good accuracy.
* It is among of the most commonly used packages in data science competition site [Kaggle](www.kaggle.com)
* The distributed version inheritates all the optimizations and scales to even larger dataset.

How to Use
====
* Input data format: LIBSVM format. The example here uses generated data in ../data folder.
* Put the data into some distribute filesytem (S3 or HDFS)
* Use tracker script in dmlc-core/tracker to submit the jobs
* Like all other DMLC tools, xgboost support taking a path to a folder as input argument
  - All the files in the folder will be used as input
* Quick start in Hadoop YARN: run ```bash run_yarn.sh <n_hadoop_workers> <n_thread_per_worker> <path_in_HDFS>```

Example
====
* [run_yarn.sh](run_yarn.sh) shows how to submit job to Hadoop via YARN.

Documentation  
====
* See [Document of xgboost](https://github.com/dmlc/xgboost/doc/README.md)
  - The distributed version is essetially the same as CLI version of xgboost
  - The only difference is change data path to path in hdfs (or s3) and submit with dmlc_yarn.py script  
* It is highly recommended to try single machine version first

Single machine vs Distributed Version
====
If you have used xgboost (single machine version) before, this section will show you how to run xgboost on hadoop with a slight modification on conf file.
* IO: instead of reading and writing file locally, we now use HDFS, put ```hdfs://``` prefix to the address of file you like to access
* File cache: ```dmlc_yarn.py``` also provide several ways to cache necesary files, including binary file (xgboost), conf file
  - ```dmlc_yarn.py``` will automatically cache files in the command line. For example, ```dmlc_yarn.py -n 3 $localPath/xgboost.dmlc mushroom.hadoop.conf``` will cache "xgboost.dmlc" and "mushroom.hadoop.conf".
  - You could also use "-f" to manually cache one or more files, like ```-f file1 -f file2```
  - The local path of cached files in command is "./".
* More details of submission can be referred to the usage of ```dmlc_yarn.py```.
* The model saved by hadoop version is compatible with single machine version.

Notes
====
* The code is optimized with multi-threading, so you will want to run xgboost with more vcores for best performance.
  - You will want to set <n_thread_per_worker> to be number of cores you have on each machine.


External Memory Version
====
XGBoost supports external memory, this will make each process cache data into local disk during computation, without taking up all the memory for storing the data.
See [external memory](https://github.com/dmlc/xgboost/doc/external_memory.md) for syntax using external memory.

You only need to add cacheprefix to the input file to enable external memory mode. For example set training data as 
```
data=hdfs:///path-to-my-data/#dtrain.cache
```
This will make xgboost more memory efficient, allows you to run xgboost on larger-scale dataset.

Contribution
====
Contribution of documents and usecases of distributed xgboost is welcomed! Open an issue at wormhole and send a pull request! 