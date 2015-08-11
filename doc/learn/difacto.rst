Factorization Machine
=====================

Difacto is refined factorization machine (FM) with sparse memory adaptive
constraints.

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

Train by Asynchronous SGD. *w* is updated via FTRL while *V* via adagrad.

Configuration
---------------------

The configure is defined in the protobuf file `config.proto <https://github.com/dmlc/wormhole/blob/master/learn/difacto/config.proto>`_

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

   float, lambda_l1, "l1 regularizer for :math:`w`: :math:`\lambda_1 |w|_1`"
   float, lambda_l2, "l2 regularizer for :math:`w`: :math:`\lambda_2 \|w\|_2^2`"
   float, lr_eta, "learning rate :math:`\eta` (or :math:`\alpha`) for :math:`w`"
   Config.Embedding, embedding, "the embedding :math:`V`"
   int32, minibatch, "the size of minibatch. the smaller, the faster the convergence, but the/ slower the system performance"
   int32, max_data_pass, "the maximal number of data passes"
   bool, early_stop, "stop earilier if the validation objective is less than  prev_obj - min_objv_decr"

Config.Embedding
``````````````````````````````````````````
embedding :math:`V`. basic:

.. csv-table::
   :header: Type, Field, Description

   int32, dim, "the embedding dimension :math:`k`"
   int32, threshold, "features with occurence &lt; threshold have no embedding (:math:`k=0`)"
   float, lambda_l2, "l2 regularizer for :math:`V`: :math:`\lambda_2 \|V_i\|_2^2`"

advanced:

.. csv-table::
   :header: Type, Field, Description

   float, init_scale, "V is initialized by uniformly random weight in/   [-init_scale, +init_scale]"
   float, dropout, "apply dropout on the gradient of :math:`V`. no in default"
   float, grad_clipping, "project the gradient of :math:`V` into :math:`[-c c]`. no in default"
   float, grad_normalization, "normalized the l2-norm of gradient of :math:`V`. no in default"
   float, lr_eta, "learning rate :math:`\eta` for :math:`V`. if not specified, then share the same with :math:`w`"
   float, lr_beta, "leanring rate :math:`\beta` for :math:`V`."

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
   float, print_sec, "print the progress every n sec during training. 1 sec in default"
   float, lr_beta, "learning rate :math:`\beta`, 1 in default"
   float, min_objv_decr, "the minimal objective decrease in early stop"
   bool, l1_shrk, "use or not use the contraint :math:`V_i = 0` if :math:`w_i = 0`. yes in default"
   int32, num_threads, "number of threads used within a worker and a server"
   int32, max_concurrency, "the maximal concurrent minibatches being processing at the same time for/ sgd, and the maximal concurrent blocks for block CD. 2 in default."
   bool, key_cache, "cache the key list on both sender and receiver to reduce communication/ cost. it may increase the memory usage"
   bool, msg_compression, "compression the message to reduce communication cost. it may increase the/ computation cost."
   int32, fixed_bytes, "convert floating-points into fixed-point integers with n bytes. n can be 1,/ 2 and 3. 0 means no compression."

Performance
-----------
