# Distributed Factorization Machine

For a *p* dimension example *x*, FM models the data by

![hat_y](guide/hat_y.png)

where *w_i* denotes by the i-th element of the *p*-length vector *w*, and *v_i*
denotes by the i-th row of the *p*-by-*k* matrix *V*.

Given training data pairs *(x,y)*, FM learns the model
*w* and *V* by solving the following objective:

<!-- \left[\sum_{(x,y)} \ell(\hat y(x,w,V), y)\right] + \lambda_1 |w|_1 + \frac{1}{2} \lambda_2
\|w\|_2^2 + \frac{1}{2} \mu_2 \|V\|_F^2 -->

![obj](guide/obj.png)

Here *ℓ* is the loss function such as logistic loss.

## Quick start

Go the root directory of `wormhole`, then build by `make difacto`. Next try
a small dataset using 2 worker and 1 server:

```
tracker/dmlc_local.py -n 2 -s 1 bin/difacto.dmlc learn/difacto/guide/demo.conf
```

## More

- [Tutorial for the Criteo Kaggle CTR competition](http://wormhole.readthedocs.org/en/latest/tutorial/criteo_kaggle.html)
- [User Guide](http://wormhole.readthedocs.org/en/latest/learn/difacto.html)
