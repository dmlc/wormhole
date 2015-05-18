# Train Criteo dataset on AWS

This is a step-by-step guide how to train a model on the
[Criteo Terabyte CTR dataset]((http://labs.criteo.com/downloads/download-terabyte-click-logs/))
on [AWS EC2](http://aws.amazon.com/ec2/).

## Prepare dataset

We first save the dataset on [S3](http://aws.amazon.com/s3/). Assume
[S3cmd](http://s3tools.org/s3cmd) has been installed in `s3cmd_path` and the
destination path on S3 is `s3_path`. The following script will download the
data, uncompress, and then upload. Each file will takes ~45GB space.

```bash
s3cmd_path=~/s3cmd-1.5.2
s3_path=s3://criteo

for i in {0..23}; do
    curl http://azuremlsampleexperiments.blob.core.windows.net/criteo/day_${i}.gz | \
        gzip -d | ${s3cmd_path}/s3cmd put - ${s3_path}/day_${i}
done
```

We can also save the data in a seekable binary format, which is ~27GB for each
file.

```bash
s3cmd_path=~/s3cmd-1.5.2
s3_path=s3://criteo
wormhole_path=~/wormhole

for i in {0..23}; do
    curl http://azuremlsampleexperiments.blob.core.windows.net/criteo/day_${i}.gz | \
        gzip -d | ${wormhole_path}/learn/linear/build/text2rec_criteo stdin stdout | \
        ${s3cmd_path}/s3cmd put - ${s3_path}/day_${i}
done
```


## Setup EC2 instance

## Runing
