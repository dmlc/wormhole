# Tutorial for the Criteo Kaggle Competition

This tutorial gives an example to train a linear model on the
[Criteo Kaggle CTR competetion dataset](https://www.kaggle.com/c/criteo-display-ad-challenge/)

Assume you are using `bash` or `sh` and currently in a working directory.

### Preparation

First build wormhole:

```
git clone https://github.com/dmlc/wormhole
cd wormhole && make linear
```

Then download the dataset and split `train.txt` to multiple parts (300MB per
part) for training and validation

```
wget https://s3-eu-west-1.amazonaws.com/criteo-labs/dac.tar.gz
tar -zxvf dac.tar.gz
mkdir data
wormhole/bin/text2crb train.txt data/train criteo 300
```

### Train

We first generate a config file for training. It uses the first 20 parts for
trainig and last 6 parts for validationg.

```
cat >train.conf <<EOF
train_data = "data/train-part_[0-1].*"
val_data = "data/train-part_2.*"
data_format = "crb"
model_out = "model/criteo"

lambda_l1 = 4
lr_eta = .1
minibatch = 10000
max_data_pass = 1
EOF
```

Then train the model in local machine with 10 workers and 10 servers.

```
mkdir model
wormhole/tracker/dmlc_local.py -n 10 -s 10 wormhole/bin/linear.dmlc train.conf
```

A sample trainig log:

```
2015-07-22 04:50:55,285 INFO start listen on 192.168.0.112:9091
connected 10 servers and 10 workers
training #iter = 1
  sec #example delta #ex    |w|_0       logloss     AUC    accuracy
    1  1.8e+06  1.8e+06        30509  0.507269  0.758684  0.769462
    2  3.7e+06  1.9e+06        50692  0.469855  0.782046  0.780102
    3  5.5e+06  1.9e+06        70856  0.462922  0.785047  0.784311
    4  7.5e+06    2e+06        85960  0.462718  0.786288  0.783614
    5  9.5e+06    2e+06       101072  0.459013  0.787837  0.786144
    6  1.1e+07  1.5e+06       111139  0.458847  0.789737  0.786385
    7  1.3e+07  2.2e+06       122207  0.462001  0.788851  0.783950
    8  1.5e+07  2.1e+06       139325  0.460539  0.789178  0.785297
    9  1.7e+07    2e+06       151402  0.458131  0.790948  0.786492
   10  1.9e+07    2e+06       161498  0.460316  0.791303  0.785415
   11  2.1e+07    2e+06       171568  0.456593  0.792061  0.787479
   12  2.3e+07  1.9e+06       181625  0.456088  0.793165  0.787807
   13  2.5e+07  1.9e+06       191691  0.457250  0.791097  0.787266
   14  2.6e+07  1.1e+06       194704  0.460679  0.792672  0.784258
   15  2.8e+07  1.8e+06       204754  0.456175  0.792998  0.787669
   16    3e+07    2e+06       212809  0.453904  0.793889  0.789362
   17  3.2e+07    2e+06       222904  0.456366  0.792361  0.787056
   18  3.4e+07    2e+06       231968  0.453590  0.793880  0.789032
   19  3.6e+07    2e+06       242017  0.454674  0.794033  0.788652
   20  3.7e+07  8.4e+05       248066  0.461133  0.791255  0.784265
validating #iter = 1
  sec #example delta #ex    |w|_0       logloss     AUC    accuracy
   30  4.6e+07  9.3e+06       248066  0.459048  0.791334  0.785863
hit max number of data passes
saving final model to model/criteo
training is done!
```

### Prediction

Generate the prediction config file

```
cat >pred.conf <<EOF
val_data = "test.txt"
data_format = "criteo_test"
model_in = "model/criteo"
pred_out = "output/criteo"
EOF
```

and do prediction using

```
mkdir output
wormhole/tracker/dmlc_local.py -n 10 -s 10 wormhole/bin/linear.dmlc pred.conf
```

Then there will be multple files in `output`, join them via

```
cat output/criteo* >pred.txt
```

Then `pred.txt` will be has the same number of lines as `test.txt`, for exmaple

```
$ wc -l test.txt pred.txt
   6042135 test.txt
   6042135 pred.txt
```

and each line of `pred.txt` will be the according *<x,w>*.
