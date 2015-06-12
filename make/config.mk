#-----------------------------------------------------
#  wormhole: the configuration compile script
#
#  This is the default configuration setup for all dmlc projects
#  If you want to change configuration, do the following steps:
#
#  - copy this file to the root of wormhole folder
#  - modify the configuration you want
#  - type make or make -j n on each of the folder
#----------------------------------------------------

# choice of compiler
export CC = gcc
export CXX = g++
export MPICXX = mpicxx
export HADOOP_HOME = /data/wttian/dev/hadoop
export JAVA_HOME=/usr/lib/jvm/java-1.7.0-openjdk-1.7.0.79-2.5.5.1.el7_1.x86_64
# whether use HDFS support during compile
USE_HDFS = 1

# whether use AWS S3 support during compile
USE_S3 = 0

# path to libjvm.so
LIBJVM=$(JAVA_HOME)/jre/lib/amd64/server

DMLC_CFLAGS=-L/data/wttian/dev/hadoop/lib/native
