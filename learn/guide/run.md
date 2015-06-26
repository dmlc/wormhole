# Run jobs

All wormhole applications can be launched in local and in multiple machines via
trackers in [dmlc-core](https://github.com/dmlc/dmlc-core/tree/master/tracker)

Assume the current directory is `wormhole/learn/linear/guide`, and we are going
to run `demo.conf` by 1 servers and 4 workers:

- launch in local machine

```bash
../../../dmlc-core/tracker/dmlc_local.py -n 4 -s 1 ../build/linear.dmlc demo.conf
```

- launch by [yarn](http://hadoop.apache.org/docs/current/hadoop-yarn/hadoop-yarn-site/YARN.html)

```bash
../../../dmlc-core/tracker/dmlc_yarn.py -n 4 -s 1 ../build/linear.dmlc demo.conf
```

- launch by [mpich2](https://www.mpich.org/)

First save all hostnames in `hosts`, then

```bash
../../../dmlc-core/tracker/dmlc_mpi.py -H hosts -n 4 -s 1 ../build/linear.dmlc demo.conf
```

(If we are using [open-mpi](http://www.open-mpi.org/), then we need to change `cmd +=
' -env %s %s' % (k, v)` to `cmd += ' -x %s' % k` in `dmlc_mpi.py` for setting
environment variables.)

Check [dmlc-core](https://github.com/dmlc/dmlc-core/tree/master/tracker) for
more launching options.

## More options for parameter server backended applications

There are several useful flags in [ps-lite](https://github.com/dmlc/ps-lite).

- `-log_dir ~/log/` ask all workers and servers to save the logfiles in
  `~/log/`. It's convenient when `~` is mounted on a NFS. Otherwise, all logs
  are available at `/tmp/`

- `-local` slightly faster when running jobs in a single machine

- `-vmodule van*=1` print all network communications by logging all information
  in `van.h` and `van.cc`. We can debug other components such as `manager.h/.cc`
  by `-vmodule manager*=1`
