![Wormhole](wormhole.png)

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
* [L-BFGS based linear solver](learn/lbfgs-linear)

Build
====
* copy ```make/config.mk``` to root folder
* modify according to your settings
* type ```make``` or ```make name-of-tool-you-want```

How to Submit Jobs
====
* make sure dmlc-core exist in root folder
  - type ```make dmlc-core``` to get it
* Use the submission script in ```dmlc-core/tracker``` to submit job to the platform of your choice

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
