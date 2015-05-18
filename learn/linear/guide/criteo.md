# Train the Criteo CTR dataset on AWS

This is a step-by-step guide on how to train a linear model on the
public [Criteo Terabyte CTR dataset](http://labs.criteo.com/downloads/download-terabyte-click-logs/)
by using [AWS EC2](http://aws.amazon.com/ec2/). This dataset contains around 4
billions examples and 800 million unique features.

## Prepare dataset

We first save the dataset on [S3](http://aws.amazon.com/s3/). Assume
[S3cmd](http://s3tools.org/s3cmd) has been installed in `s3cmd_path` and the
destination path on S3 is `s3_path`. The following script will download,
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
        gzip -d | ${wormhole_path}/learn/linear/build/text2rec_criteo stdin stdout | \
        ${s3cmd_path}/s3cmd put - ${s3_path}/day_${i}
done
```

## Setup EC2 instance

Here we give a quick solution based on `NFS` and `mpirun`. However, a cluster
resource manager such as `Yarn` is much better if the cluster is shared among
several users.

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
make -j8
```

(try `make` again if there no `build/async_sgd`)

Now we can test wormhole the by using `demo_local.sh` or the examples in next
section without the hostfile (namely no `-H ./hosts`).

Then we setup
[NFS](https://help.ubuntu.com/lts/serverguide/network-file-system.html) to
distribute the binary file and configure file, and `mpirun` to launch the jobs.

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

Link the tracker
```
ln -s ../../../dmlc-core/tracker/dmlc_mpi.py .
```

Now we can launch the job. For example, using 100 workers and 100 servers to
train a sparse logistic regression using asynchronous SGD.

```bash
~/wormhole/learn/linear/guide $ ./dmlc_mpi.py -s 100 -n 100 -H hosts ../build/async_sgd criteo_s3.conf
2015-05-18 21:17:35,928 INFO start listen on 172.30.0.221:9091
waiting 100 workers and 100 servers are connected
training #iter = 0
time(sec)  #example  delta #ex    |w|_1       objv       AUC    accuracy
      2       5e+06   5000000     381946  0.149147  0.641368  0.967909
      3    1.01e+07   5100000    1042058  0.133591  0.716495  0.967949
      4    2.09e+07  10800000    3935831  0.357829  0.722713  0.854518
      5       3e+07   9100000    5775722  0.608035  0.643276  0.940628
      6     4.1e+07  11000000    6987563  1.200063  0.555872  0.967999
      7    5.12e+07  10200000    8114344  1.267120  0.553158  0.964398
      8     6.1e+07   9800000    9136571  1.187503  0.559613  0.967896

```

## Performance

Using 5 EC2 c4.8x machines with 100 workers, 100 servers and the default
`criteo_s3.conf` (minibatch size = 100K and max delay = 4), it processes 9.5
million examples per second. One pass of the data (which is good enough) costs
around 7 minutes.
