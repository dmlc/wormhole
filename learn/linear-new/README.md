# Linear Method

Given data pairs *(x,y)*, the linear method learn the model vector *w* by minizing the
following objective function

![obj](guide/obj.png)

where *ℓ* is the loss function such as logistic loss and hinge loss.


## Build and Run

First run `../../make/build_deps.sh` if you didn't run it before. Then
`make`. (Only tested on linux with gcc >= 4.8).

Train a small dataset in local machine by 1 worker and 1 server:

```
cd guide
../../../dmlc-core/tracker/dmlc_local.py -n 1 -s 1 ../build/fm.dmlc demo.conf
```

More documents:

- [Use bigger datasets](../guide/data.md)
- [Launch jobs in multiple machines](../guide/run.md)
