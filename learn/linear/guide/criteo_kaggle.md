```

wget https://s3-eu-west-1.amazonaws.com/criteo-labs/dac.tar.gz
tar -zxvf dac.tar.gz
mkdir data
${wormhole}/bin/text2crb train.txt data/train criteo 300

mkdir model
cp ${wormhole}/learn/linear/guide/criteo_kaggle.conf .
${wormhole}/tracker/dmlc_local.py -n 10 -s 10 .${wormhole}/bin/linear.dmlc criteo_kaggle.conf
```
