import sys
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
flags.DEFINE_integer("worker_index", 0,
                     "Worker task index, should be >= 0. worker_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")

flags.DEFINE_string("workers", None,
                    "The worker url list, separated by comma (e.g. tf-worker1:2222,1.2.3.4:2222)")

flags.DEFINE_string("parameter_servers", None,
                    "The ps url list, separated by comma (e.g. tf-ps2:2222,1.2.3.5:2222)")

flags.DEFINE_string("worker_grpc_url", None,
                    "Worker GRPC URL (e.g., grpc://1.2.3.4:2222, or "
                    "grpc://tf-worker0:2222)")

flags.DEFINE_string("name_scope", None, "The variable name scope.")
FLAGS = flags.FLAGS

TRAING_STEP = 5000
hidden_nodes = 500

def nn_layer(input_tensor, input_dim, output_dim, act=tf.nn.relu):
    with tf.name_scope(FLAGS.name_scope):  
        weights = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1, seed = 2))
        biases = tf.Variable(tf.constant(0.1, shape=[output_dim]))
        activations = act(tf.matmul(input_tensor, weights) + biases)
    return activations

def model(x, y_, global_step):   
    hidden1 = nn_layer(x, 784, hidden_nodes)
    y = nn_layer(hidden1, hidden_nodes, 10, act=tf.nn.softmax)
        
    cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy, global_step=global_step)
        
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    return train_step, accuracy
    
print("Loading data from worker index = %d" % FLAGS.worker_index)
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
print("Testing set size: %d" % len(mnist.test.images))
print("Training set size: %d" % len(mnist.train.images))

print("Worker GRPC URL: %s" % FLAGS.worker_grpc_url)
print("Workers = %s" % FLAGS.workers)
print("Using name scope %s" % FLAGS.name_scope)

is_chief = (FLAGS.worker_index == 0)
cluster = tf.train.ClusterSpec({"ps": FLAGS.parameter_servers.split(","), "worker": FLAGS.workers.split(",")})
# Construct device setter object
device_setter = tf.train.replica_device_setter(cluster=cluster)
        
# The device setter will automatically place Variables ops on separate
# parameter servers (ps). The non-Variable ops will be placed on the workers.
with tf.device(device_setter):
    with tf.name_scope(FLAGS.name_scope):
        global_step = tf.Variable(0, trainable=False)
 
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    val_feed = {x: mnist.test.images, y_: mnist.test.labels}
            
    train_step, accuracy = model(x, y_, global_step)
        
    sv = tf.train.Supervisor(is_chief=is_chief,
                             logdir="/tmp/dist-mnist-log/train",
                             saver=tf.train.Saver(),
                             init_op=tf.initialize_all_variables(),
                             recovery_wait_secs=1,
                             global_step=global_step)
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True,
                                 device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.worker_index])
        
    # The chief worker (worker_index==0) session will prepare the session,
    # while the remaining workers will wait for the preparation to complete.
    if is_chief:
        print("Worker %d: Initializing session..." % FLAGS.worker_index)
    else:
        print("Worker %d: Waiting for session to be initialized..." % FLAGS.worker_index)
        
    with sv.prepare_or_wait_for_session(FLAGS.worker_grpc_url, config=sess_config) as sess:
        print("Worker %d: Session initialization complete." % FLAGS.worker_index)
            
        # Perform training
        time_begin = time.time()
        print("Training begins @ %f" % time_begin)
    
        local_step = 0
        while True:
            # Training feed
            batch_xs, batch_ys = mnist.train.next_batch(100)
            train_feed = {x: batch_xs, y_: batch_ys}

            _, step = sess.run([train_step, global_step], feed_dict=train_feed)
            if local_step % 100 == 0:
                print("Worker %d: training step %d done (global step: %d); Accuracy: %g" % 
                      (FLAGS.worker_index, local_step, step, sess.run(accuracy, feed_dict=val_feed)))
            if step >= TRAING_STEP: break
            local_step += 1
    
        time_end = time.time()
        print("Training ends @ %f" % time_end)
        training_time = time_end - time_begin
        print("Training elapsed time: %f s" % training_time)
    
        # Accuracy on test data
        print("Final test accuracy = %g" % (sess.run(accuracy, feed_dict=val_feed)))

