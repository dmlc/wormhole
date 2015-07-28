# Protocol Documentation
<a name="top"/>

## Table of Contents
* [config.proto](#config.proto)
 * [Config](#dmlc.linear.Config)
 * [Config.Algo](#dmlc.linear.Config.Algo)
 * [Config.Loss](#dmlc.linear.Config.Loss)
* [Scalar Value Types](#scalar-value-types)

<a name="config.proto"/>
<p align="right"><a href="#top">Top</a></p>

## config.proto

<a name="dmlc.linear.Config"/>
### Config


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| train_data | [string](#string) | optional | Both train_data and val_data can be either a directory or a wildcard/ filename such as &quot;part-[0-9].*&quot;. also support hdfs:/// and s3:/// |
| val_data | [string](#string) | optional |  |
| data_format | [string](#string) | optional | data format. support libsvm, crb, criteo, adfea, ... |
| model_out | [string](#string) | optional | model output |
| save_model | [int32](#int32) | optional | save model for every *save_model* data pass. if 0, then only save the last/ one |
| model_in | [string](#string) | optional | model input |
| load_model | [int32](#int32) | optional | load model from iter = *load_model*. if 0, then load the last one |
| pred_out | [string](#string) | optional | output of the prediction results. if specified, then do/ prediction. otherwise run the training job. |
| loss | [Config.Loss](#dmlc.linear.Config.Loss) | optional |  |
| lambda_l1 | [float](#float) | optional | elastic-net regularizer:/   lambda_l1 * ||w||_1 + .5 * lambda_l2 * ||w||_2^2 |
| lambda_l2 | [float](#float) | optional |  |
| algo | [Config.Algo](#dmlc.linear.Config.Algo) | optional |  |
| minibatch | [int32](#int32) | optional | the size of minibatch for minibatch SGD. a small value improves the/ convergence rate, but may decrease the system performance |
| max_data_pass | [int32](#int32) | optional | the maximal number of data passes |
| lr_eta | [float](#float) | optional | learning rate, often in the format lr_eta / (lr_beta + x), where x depends/ on the updater, such as sqrt(iter), or the cumulative gradient on adagrad |
| use_worker_local_data | [bool](#bool) | optional | each worker reads the data in local if the data have been/ dispatched into workers' local disks. it can reduce the cost to access data/ remotely |
| num_parts_per_file | [int32](#int32) | optional | virtually partition a file into nparts for better loadbalance. |
| rand_shuffle | [int32](#int32) | optional | randomly shuffle data for minibatch SGD. a minibatch is randomly pick from/ rand_shuffle * minibatch_size examples. |
| neg_sampling | [float](#float) | optional | down sampling negative examples in the training data |
| disp_itv | [float](#float) | optional | print the progress every n sec |
| lr_beta | [float](#float) | optional | learning rate, see lr_eta |
| num_threads | [int32](#int32) | optional | number of threads used within a worker and a server |
| max_delay | [int32](#int32) | optional | maximal allowed delay during synchronization. it is the maximal number of/ parallel minibatches for SGD, and parallel block for block CD. |
| key_cache | [bool](#bool) | optional | cache the key list on both sender and receiver to reduce communication/ cost. it may increase the memory usage |
| msg_compression | [bool](#bool) | optional | compression the message to reduce communication cost. it may increase the/ computation cost. |
| fixed_bytes | [int32](#int32) | optional | convert floating-points into fixed-point integers with n bytes. n can be 1,/ 2 and 3. 0 means no compression. |


<a name="dmlc.linear.Config.Algo"/>
### Config.Algo
the learning  method

| Name | Number | Description |
| ---- | ------ | ----------- |
| SGD | 1 | minibatch SGDstandard sgd |
| ADAGRAD | 2 | adaptive gradient descent |
| FTRL | 3 |  |

<a name="dmlc.linear.Config.Loss"/>
### Config.Loss
the loss function

| Name | Number | Description |
| ---- | ------ | ----------- |
| SQUARE | 1 |  |
| LOGIT | 2 |  |
| SQUARE_HINGE | 4 | HINGE = 3; |


<a name="scalar-value-types"/>
## Scalar Value Types

| .proto Type | Notes | C++ Type | Java Type | Python Type |
| ----------- | ----- | -------- | --------- | ----------- |
| <a name="double"/> double |  | double | double | float |
| <a name="float"/> float |  | float | float | float |
| <a name="int32"/> int32 | Uses variable-length encoding. Inefficient for encoding negative numbers – if your field is likely to have negative values, use sint32 instead. | int32 | int | int |
| <a name="int64"/> int64 | Uses variable-length encoding. Inefficient for encoding negative numbers – if your field is likely to have negative values, use sint64 instead. | int64 | long | int/long |
| <a name="uint32"/> uint32 | Uses variable-length encoding. | uint32 | int | int/long |
| <a name="uint64"/> uint64 | Uses variable-length encoding. | uint64 | long | int/long |
| <a name="sint32"/> sint32 | Uses variable-length encoding. Signed int value. These more efficiently encode negative numbers than regular int32s. | int32 | int | int |
| <a name="sint64"/> sint64 | Uses variable-length encoding. Signed int value. These more efficiently encode negative numbers than regular int64s. | int64 | long | int/long |
| <a name="fixed32"/> fixed32 | Always four bytes. More efficient than uint32 if values are often greater than 2^28. | uint32 | int | int |
| <a name="fixed64"/> fixed64 | Always eight bytes. More efficient than uint64 if values are often greater than 2^56. | uint64 | long | int/long |
| <a name="sfixed32"/> sfixed32 | Always four bytes. | int32 | int | int |
| <a name="sfixed64"/> sfixed64 | Always eight bytes. | int64 | long | int/long |
| <a name="bool"/> bool |  | bool | boolean | boolean |
| <a name="string"/> string | A string must always contain UTF-8 encoded or 7-bit ASCII text. | string | String | str/unicode |
| <a name="bytes"/> bytes | May contain any arbitrary sequence of bytes. | string | ByteString | str |
