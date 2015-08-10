Linear Method
=============

Given data pairs :math:`(x,y)`, the linear method learns the model vector
:math:`w` by minizing the following objective function:

.. math::
   \sum_{(x,y)} \ell(y, \langle x, w \rangle) + \lambda_1 |w|_1 + \lambda_2 \|w\|_2^2

where :math:`\ell(y, p)` is the loss function, see :ref:`config.loss`.

Configuration
---------------------

The configuration is defined in the protobuf file `config.proto <https://github.com/dmlc/wormhole/blob/master/learn/linear/config.proto>`_

Input & Output
~~~~~~~~~~~~~~~~~~~

.. csv-table::
   :header: Type, Field, Description

   string, train_data, "The training data, can be either a directory or a wildcard filename"
   string, val_data, "The validation or test data, can be either a directory or a wildcard filename"
   string, data_format, "data format. supports libsvm, crb, criteo, adfea, ..."
   string, model_out, "model output filename"
   string, model_in, "model input filename"
   string, predict_out, "the filename for prediction output. if specified, then run/ prediction. otherwise run training"


Model and Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table::
   :header: Type, Field, Description

   Config.Loss, loss, "the loss function. default is LOGIT"
   float, lambda_l1, "l1 regularizer: :math:`\lambda_1 |w|_1`"
   float, lambda_l2, "l2 regularizer: :math:`\lambda_2 \|w\|_2^2`"
   Config.Algo, algo, "the learning method, default is FTRL"
   int32, minibatch, "the size of minibatch. the smaller, the faster the convergence, but the/ slower the system performance"
   int32, max_data_pass, "the maximal number of data passes"
   float, lr_eta, "the learning rate :math:`\eta` (or :math:`\alpha`). often uses the largest/ value when not diverged"

.. _config.loss:

Config.Loss
``````````````
.. csv-table::
   :header: Name, Description

   SQUARE, "square loss: :math:`\frac12 (p-y)^2`"
   LOGIT, "logistic loss: :math:`\log(1+\exp(-yp))`"
   SQUARE_HINGE, "squared hinge loss: :math:`\max\left(0, (1-yp)^2\right)`"

Config.Algo
``````````````

.. csv-table::
   :header: Name, Description

   SGD, "asynchronous minibatch SGD"
   ADAGRAD, "similar to SGD, but use adagrad"
   FTRL, "similar to ADAGRAD, but use FTRL for better sparsity"

Adavanced Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table::
   :header: Type, Field, Description

   int32, save_iter, "save model for every k data pass. default is -1, which only saves for the/ last iteration"
   int32, load_iter, "load model from the k-th iteration. default is -1, which loads the last/ iteration model"
   bool, local_data, "give a worker the data only if it can access. often used when the data has/ been dispatched to workers' local filesystem"
   int32, num_parts_per_file, "virtually partition a file into n parts for better loadbalance. default is 10"
   int32, rand_shuffle, "randomly shuffle data for minibatch SGD. a minibatch is randomly picked from/ rand_shuffle * minibatch examples. default is 10."
   float, neg_sampling, "down sampling negative examples in the training data. no in default"
   bool, prob_predict, "if true, then outputs a probability prediction. otherwise :math:`\langle  x, y \rangle`"
   float, dropout, "the probably to set a gradient to 0. no in default"
   float, print_sec, "print the progress every n sec during training. 1 sec in default"
   float, lr_beta, "learning rate :math:`\beta`, 1 in default"
   int32, num_threads, "number of threads used by a worker / a server. 2 in default"
   int32, max_concurrency, "the maximal concurrent minibatches being processing at the same time for/ sgd, and the maximal concurrent blocks for block CD. 2 in default."
   bool, key_cache, "cache the key list on both sender and receiver to reduce communication/ cost. it may increase the memory usage"
   bool, msg_compression, "compression the message to reduce communication cost. it may increase the/ computation cost."
   int32, fixed_bytes, "convert floating-points into fixed-point integers with n bytes. n can be 1,/ 2 and 3. 0 means no compression."

Performance
-----------
