# Train the Criteo CTR dataset on AWS

**NOTICE: this file is out of date. will be updated soon**

This is a step-by-step guide on how to train a linear model on the
public [Criteo Terabyte CTR dataset](http://labs.criteo.com/downloads/download-terabyte-click-logs/)
by using [AWS EC2](http://aws.amazon.com/ec2/). This dataset contains around 4
billions examples and 800 million unique features.

## Prepare dataset

We first put the dataset into [S3](http://aws.amazon.com/s3/). (It is similar to
put data into HDFS by changing `s3cmd put` to `hadoop fs -put`.)
Assume [S3cmd](http://s3tools.org/s3cmd) has been installed in `s3cmd_path` and
the destination path on S3 is `s3_path`. The following script will download,
uncompress, and then upload the data. Each file will take ~45GB space.

```bash
s3cmd_path=~/s3cmd-1.5.2
s3_path=s3://ctr-data/criteo

for i in {0..23}; do
    curl http://azuremlsampleexperiments.blob.core.windows.net/criteo/day_${i}.gz | \
        gzip -d | ${s3cmd_path}/s3cmd put - ${s3_path}/day_${i}
done
```

We can also save the data in a seekable binary format, which is ~27GB for each
file.

```bash
s3cmd_path=~/s3cmd-1.5.2
s3_path=s3://ctr-data/criteo
wormhole_path=~/wormhole

for i in {0..23}; do
    curl http://azuremlsampleexperiments.blob.core.windows.net/criteo/day_${i}.gz | \
        gzip -d | ${wormhole_path}/learn/linear/build/text2rec stdin stdout criteo | \
        ${s3cmd_path}/s3cmd put - ${s3_path}/day_${i}
done
```

## Setup EC2 instance

We need a EC2 cluster to run the job. If the cluster has already been created
and setup with any resource manager such as `Yarn`, `Sun Grid Engine` or simply
`mpirun`, we can skip most of the content of this section except for building
`wormhole`. Otherwise, we will give a quick tutorial on how to starting a
cluster from scratch.

### Setup the master node

Launch an Ubuntu 14.04 instance as the master node, and install `gcc`.

```bash
sudo apt-get update && sudo apt-get install -y build-essential git libcurl4-openssl-dev
```

Next build wormhole with S3 support

```bash
git clone https://github.com/dmlc/wormhole.git
cd ~/wormhole/learn/linear
./build_deps.sh
cd ../../
deps_path=`pwd`/repo/ps-lite/deps
cat <<< "# sample config for learn/linear
CC = gcc
CXX = g++
USE_GLOG = 1
DEPS_PATH = $deps_path
STATIC_DEPS = 1
USE_HDFS = 0
USE_S3 = 1
USE_KEY32 = 1" >config.mk
cd learn/linear
make
```

(try `make` again if there no `build/async_sgd`)

Now we can test wormhole the by using `demo_local.sh` or the examples in next
section without the hostfile (namely no `-H ./hosts`).

Then we setup
[NFS](https://help.ubuntu.com/lts/serverguide/network-file-system.html) to
distribute the binary file and configure file, and use `mpirun` to launch the jobs.

```bash
sudo apt-get install nfs-kernel-server mpich2
echo "/home/ubuntu/  *(rw,sync,no_subtree_check)" | sudo tee /etc/exports
sudo service nfs-kernel-server start
```

Finally copy the `pem` file used to access this machine to `~/.ssh/id_rsa` so that
it can ssh to all slaves machines.


### Setup the slave nodes

The slaves nodes are used for the actual computations. After creating them using the
same key pair used for the master node, save their private IPs in a file, such
as

```bash
~/wormhole/learn/linear/guide $ cat hosts
172.30.0.172
172.30.0.171
172.30.0.170
172.30.0.169
172.30.0.168
```

Setup `NFS` and `mpirun` on them. Assume the master node has private IP `172.30.0.221`

```bash
while read h; do
    echo $h
    ssh -o StrictHostKeyChecking=no $h <<'ENDSSH'
sudo apt-get update
sudo apt-get install -y nfs-common mpich2
sudo mount 172.30.0.221:/home/ubuntu /home/ubuntu
ENDSSH
done <hosts
```

Now we can lunch jobs via `mpirun`. For example,

```bash
~/wormhole/learn/linear/guide $ mpirun -hostfile hosts pwd
/home/ubuntu/wormhole/learn/linear/guide
/home/ubuntu/wormhole/learn/linear/guide
/home/ubuntu/wormhole/learn/linear/guide
/home/ubuntu/wormhole/learn/linear/guide
/home/ubuntu/wormhole/learn/linear/guide
```

Finally install `gcc` on all master nodes
```bash
mpirun -hostfile hosts sudo apt-get install -y build-essential libcurl4-openssl-dev
```

## Run

We first need to set the access key properly. Add the following two lines in
`~/.bashrc`
```bash
export AWS_ACCESS_KEY_ID=YOUR_ID
export AWS_SECRET_ACCESS_KEY=YOUR_KEY
```

Change [criteo_s3.conf](./criteo_s3.conf) properly. If we use the raw text file, use
```bash
train_data = "s3://ctr-data/criteo/day_.*"
data_format = "criteo"
```
or the recordio binary file
```bash
train_data = "s3://ctr-data/criteo/day_.*.rec"
data_format = "criteo_rec"
```

If the cluster has been setup with `mpirun`, we can use `dmlc_mpi.py` to launch
the job. Otherwise use other trackers such as `dmlc_yarn.py` in [dmlc-core](https://github.com/dmlc/dmlc-core/tree/master/tracker)

Link the tracker
```
ln -s ../../../dmlc-core/tracker/dmlc_mpi.py .
```

Here is the example to launch 100 workers and 100 servers to
train a sparse logistic regression using asynchronous SGD.

```
2015-05-20 03:18:24,248 INFO start listen on 172.30.0.221:9091
waiting 80 workers and 80 servers are connected
training #iter = 0
time(sec)  #example  delta #ex    |w|_1     objv       AUC    accuracy
      2       7e+05    700000          0  0.462639  0.518272  0.967187
      3     5.2e+06   4500000     432809  0.442459  0.532980  0.886363
      4    1.51e+07   9900000    1815484  0.573114  0.558478  0.936006
      5    2.58e+07  10700000    2828646  0.661016  0.577930  0.962926
      6    4.05e+07  14700000    3323027  0.703975  0.609182  0.967786
      7    5.12e+07  10700000    3857668  0.661458  0.647286  0.967767
      8    6.19e+07  10700000    4361416  0.550164  0.664641  0.967444
      9    7.24e+07  10500000    4735214  0.426738  0.682694  0.967390
     10    8.05e+07   8100000    4945155  0.331575  0.691858  0.965669
     11    9.06e+07  10100000    5044148  0.274531  0.702081  0.961745
     12   1.035e+08  12900000    5099013  0.222166  0.712238  0.960809
     ...
     90  8.4013e+08   9263320    5454075  0.132056  0.769849  0.965639
     91  8.4912e+08   8988195    5457934  0.130898  0.770375  0.965987
     92  8.5792e+08   8808806    5462317  0.129705  0.771240  0.966339
     93   8.677e+08   9778023    5466549  0.128795  0.772680  0.966592
     94  8.7717e+08   9470611    5470569  0.127350  0.773038  0.967028
     95  8.8657e+08   9395147    5474914  0.125767  0.772082  0.967455
     96  8.9523e+08   8661666    5478443  0.125678  0.773303  0.968082
     97  9.0396e+08   8728920    5482195  0.125572  0.772817  0.967699
     98  9.1412e+08  10161781    5486015  0.124886  0.771171  0.967879
     99  9.2162e+08   7500000    5490156  0.125723  0.772345  0.967641
    100  9.3152e+08   9900000    5494079  0.125193  0.771567  0.967857
    101  9.3802e+08   6500000    5497386  0.124494  0.772659  0.968086
    102  9.4592e+08   7900000    5500650  0.126277  0.771401  0.967493
```


## Performance

Using 5 EC2 c4.8x machines with 100 workers, 100 servers and the default
`criteo_s3.conf` (minibatch size = 100K and max delay = 4), it processes 9.5
million examples per second. One pass of the data (which is good enough) costs
around 7 minutes.
