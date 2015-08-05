Linear Method
=============

Given data pairs :math:`(x,y)`, the linear method learns the model vector
:math:`w` by minizing the following objective function:

.. math::
   \sum_{(x,y)} \ell(y, \langle x, w \rangle) + \lambda_1 |w|_1 + \lambda_2 \|w\|_2^2

where :math:`\ell(y, p)` is the loss function. Wormhole currently supports the
following loss functions, with logistic loss being the default:

================== ========
Loss               Function
================== ========
Logistic           :math:`\log(1+\exp(-yp))`
Squared            :math:`\frac12 (p-y)^2`
Squared Hinge Loss :math:`\max\left(0, (1-yp)^2\right)`
================== ========
