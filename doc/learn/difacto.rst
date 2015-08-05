Factorization Machine
=====================

Difacto is refined factorization machine (FM) with sparse memory adaptive
constraints.

Model
-----

Given an example :math:`x \in \mathbb{R}^d` and an embedding
dimension  :math:`k`, FM models the example by

.. math::
  f(x) = \langle w, x \rangle + \frac{1}{2} \|V x\|_2^2 - \sum_{i=1}^d x_i^2 \|V_i\|^2_2

where :math:`w \in \mathbb{R}^d` and :math:`V \in \mathbb{R}^{d \times k}`
are the models we need to learn. The learning objective function is

.. math::
   \frac 1{|X|}\sum_{(x,y)} \ell(f(x), y)+ \lambda_1 |w|_1 +
   \frac12 \sum_{i=1}^d \left[\lambda_i w_i^2 + \mu_i \|V_i\|^2\right]

where the first sparse regularizer :math:`\lambda_1 |w|_1` induces a sparse
:math:`w`, while the second term is a frequency adaptive regularization, which
places large penalties for more frequently features.

Furthermore, Difacto adds two heuristics constraints

- :math:`V_i = 0` if :math:`w_i = 0`, namely we mark the embedding for feature *i*
  is inactive if the according linear term is filtered out by the sparse
  regularizer. (You can disable it by ``l1_shrk = false``)

- :math:`V_i = 0` if the occur of feature i is less the a threshold. In other
  words, Difacto does not learn an embedding for tail features. (You can specify
  the threshold via ``threshold = 10``)

Optimization method
-------------------

Asynchronous SGD. *w* is updated via FTRL while *V* via adagrad.

Configuration
-------------

The configure is defined in the protobuf file `config.proto <https://github.com/dmlc/wormhole/blob/master/learn/difacto/config.proto>`_

Performance
-----------

What's Next
-----------
