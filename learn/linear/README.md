# Linear Method

Given data pairs *(x,y)*, the linear method learns the model vector *w* by
minizing the following objective function:

![obj](guide/obj.png)

where *ℓ* is the loss function such as logistic loss and hinge loss.

## Quick start

Go the root directory of `wormhole`, then build by `make linear`. Next try
a small dataset using 2 worker and 1 server:

```
tracker/dmlc_local.py -n 2 -s 1 bin/linear.dmlc learn/linear/guide/demo.conf
```

## More

- [Tutorial for the Criteo Kaggle CTR competition](http://wormhole.readthedocs.org/en/latest/tutorial/criteo_kaggle.html)

- [User Guide](http://wormhole.readthedocs.org/en/latest/learn/linear.html)
