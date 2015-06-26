# Data

## Supported Filesystems

Wormholes can read and write data in major filesystems, such as local disk,
[HDFS](https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-hdfs/HdfsUserGuide.html),
and [S3](http://aws.amazon.com/s3/). We only need to add the prefix `hdfs://`
for HDFS files, and prefix `s3://` for S3 files. For the latter, we also need to
set the AWS account environment variables. Such as adding the following in
`~/.bashrc`:

```bash
export AWS_ACCESS_KEY_ID=YOUR_ID export
AWS_SECRET_ACCESS_KEY=YOUR_KEY
```

## Dataset formats

Wormhole is able to read various dataset formats, including widely used
[CSV](https://en.wikipedia.org/wiki/Comma-separated_values) and
[libsvm](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/), and specific
data format such as
[criteo](https://www.kaggle.com/c/criteo-display-ad-challenge/data). Adding more
data formats is also straightforward, see examples in
[minibatch_iter.h](../base/minibatch_iter.h).

The dataset can be stored either in **plain text** format or a more compact
**recordio** format. See examples below on how to convert the text format data into
the recordio format.


## Data preparation

Most wormhole applications can process the datasets with minimal
preparation. For example, sparse dataset only stores the nonzero entries by
`feature_id value`. The feature ID can be either a `string` or a `64-bit
integer`. There is no need to map these feature IDs into continuous small
integers.

## More

### Common options

If the number of works is less than the number of files. We can virtually split
the files so that each worker can process at least one file by:

```
num_parts_per_file = 10
```

The above splits each file into 10 parts with zero overhead.

In same case we can dispatch the data into machines first (assume 100 files and
10 machines, then copy 10 files to each machine's local filesystem) or reuse the
data cache of the previous job, so that we can eliminate the overhead to read
the data from distributed/network-shared filesystem. We can use
`use_worker_local_data = true` to ask each worker only read the files from it's
own directory. Such as

```
train_data = /data/part.*
use_worker_local_data = true
```


### Sample scripts

#### Put [Criteo Terabyte CTR dataset](http://labs.criteo.com/downloads/download-terabyte-click-logs/) to S3

(change `s3cmd put` to `hadoop fs -put` for HDFS)

```bash
s3cmd_path=~/s3cmd-1.5.2
s3_path=s3://ctr-data/criteo

for i in {0..23}; do
    curl http://azuremlsampleexperiments.blob.core.windows.net/criteo/day_${i}.gz | \
        gzip -d | ${s3cmd_path}/s3cmd put - ${s3_path}/day_${i}
done
```

The configuration to read the data

```bash
train_data = "s3://ctr-data/criteo/day_.*"
data_format = "criteo"
```

#### Use the recordio format instead of plain text

```bash
s3cmd_path=~/s3cmd-1.5.2
s3_path=s3://ctr-data/criteo
wormhole_path=~/wormhole

for i in {0..23}; do
    curl http://azuremlsampleexperiments.blob.core.windows.net/criteo/day_${i}.gz | \
        gzip -d | ${wormhole_path}/learn/base/text2rec stdin stdout criteo | \
        ${s3cmd_path}/s3cmd put - ${s3_path}/day_${i}
done
```

now the configuration for reading is:

```bash
train_data = "s3://ctr-data/criteo/day_.*.rec"
data_format = "criteo_rec"
```

#### Split and shuffle the Criteo dataset:

```bash
in_dir=criteo
out_dir=criteo-rec
wormhole_path=~/wormhole
nline_per_part=10000000

mkdir -p $out_dir

for i in {0..23}; do
    tmp=$out_dir/tmp
    mkdir -p $tmp
    zcat ${in_dir}/day_${i}.gz | split -l $nline_per_part - $tmp/out_

    j=0
    for o in $tmp/out_*; do
        out=${out_dir}/day_`printf %02d $i`-part_`printf %02d $j`.rec
        echo $out
        cat $o | shuf | ${wormhole_path}/learn/base/text2rec stdin $out criteo
        ((++j))
    done
    rm -r $tmp
done
```
