## Distributed TensorFlow examples.
This directory includes yaml files  several distributed TensorFlow examples.

#### Create yaml to start TensorFlow servers
This [script](https://github.com/caicloud/tensorflow-demo/blob/master/distributed/create_tf_server_yaml.py) can generate yaml file needed to create distributed tensorflow cluster. You can use ```--num_workers``` to specify number of workers and use ```--num_parameter_servers``` to specify number of parameter servers.


#### Start examples using the TensorFlow servers
This [script](https://github.com/caicloud/tensorflow-demo/blob/master/distributed/start_tf.sh) can run all the following examples on distributed TensorFlow. Example cmd:
```
./start_tf.sh 8 3 mnist_cnn.py
```
The first parameter gives the number of workers (this can be different from the nubmer of workers specified when creating the cluster).

The second parameter gives the number of parameter servers, this must be the same as num_parameter_servers specified when creating the TensorFlow cluster.

The third parameter gives the code to be run.

#### MNIST examples
- DNN example ([code](https://github.com/caicloud/tensorflow-demo/blob/master/distributed/mnist_dnn.py))
- CNN example ([code](https://github.com/caicloud/tensorflow-demo/blob/master/distributed/mnist_cnn.py))

#### Word to Vector example
- Word to Vector example ([code](https://github.com/caicloud/tensorflow-demo/blob/master/distributed/word2vector.py))

