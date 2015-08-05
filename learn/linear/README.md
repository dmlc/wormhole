# Linear Method

Given data pairs *(x,y)*, the linear method learns the model vector *w* by
minizing the following objective function:

![obj](guide/obj.png)

where *ℓ* is the loss function such as logistic loss and hinge loss.

- To build linear, first to go the root directory of `wormhole`, then build by
  `make linear`.

- Try a small dataset in local machine by 1 worker and 1 server:

```
tracker/dmlc_local.py -n 1 -s 1 bin/linear.dmlc learn/linear/guide/demo.conf
```

- [User Document](http://wormhole.readthedocs.org/en/latest/learn/linear.html)
- [Tutorial on using the Criteo Kaggle CTR competition dataset](guide/criteo_kaggle.md)
