import sys
import time
import datetime

import numpy
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
flags.DEFINE_string("data_dir", "/tmp/data",
                    "Directory for storing mnist data")
        
flags.DEFINE_boolean("download_only", False,
                     "Only perform downloading of data; Do not proceed to "
                     "session preparation, model definition or training")

flags.DEFINE_integer("worker_index", 0,
                     "Worker task index, should be >= 0. worker_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")

flags.DEFINE_string("workers", None,
                    "The worker url list, separated by comma (e.g. tf-worker1:2222,1.2.3.4:2222)")

flags.DEFINE_string("parameter_servers", None,
                    "The ps url list, separated by comma (e.g. tf-ps2:2222,1.2.3.5:2222)")

flags.DEFINE_integer("grpc_port", 2222,
                     "TensorFlow GRPC port")

flags.DEFINE_integer("train_steps", 10000,
                     "Number of (global) training steps to perform")

flags.DEFINE_string("worker_grpc_url", None,
                    "Worker GRPC URL (e.g., grpc://1.2.3.4:2222, or "
                    "grpc://tf-worker0:2222)")
FLAGS = flags.FLAGS

BATCH_SIZE = 64
EVAL_SIZE = 10000
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10
cur_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
  
print("Loading data from worker index = %d" % FLAGS.worker_index)

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
print("Testing set size: %d" % len(mnist.test.images))
print("Training set size: %d" % len(mnist.train.images))
if FLAGS.download_only: sys.exit(0)

print("Worker GRPC URL: %s" % FLAGS.worker_grpc_url)
print("Workers = %s" % FLAGS.workers)

is_chief = (FLAGS.worker_index == 0)
cluster = tf.train.ClusterSpec({"ps": FLAGS.parameter_servers.split(","), "worker": FLAGS.workers.split(",")})
# Construct device setter object
device_setter = tf.train.replica_device_setter(cluster=cluster)
        
# The device setter will automatically place Variables ops on separate
# parameter servers (ps). The non-Variable ops will be placed on the workers.
with tf.device(device_setter):
    with tf.name_scope(cur_time):
        global_step = tf.Variable(0, trainable=False)
            
        # The variables below hold all the trainable weights. 
        # Convolutional layers.
        conv1_weights = tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, 32], stddev=0.1))
        conv1_biases = tf.Variable(tf.zeros([32]))
        
        # fully connected, depth 200
        fc1_weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE // 2 * IMAGE_SIZE // 2 * 32, 200], stddev=0.1))
        fc1_biases = tf.Variable(tf.constant(0.1, shape=[200]))
        
        fc2_weights = tf.Variable(tf.truncated_normal([200, NUM_LABELS], stddev=0.1))
        fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))
    
    x = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    y_ = tf.placeholder(tf.float32, shape=(None, NUM_LABELS))

    def model(data, train=False):
        """The Model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        conv = tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        # Bias and rectified linear non-linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                
        # Reshape the feature map into a 2D matrix to feed it to the fully connected layers.
        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(pool, [-1, pool_shape[1] * pool_shape[2] * pool_shape[3]])
        
        # Fully connected layer. Note that the '+' operation automatically broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        if train: hidden = tf.nn.dropout(hidden, 0.5)
        return tf.nn.softmax(tf.matmul(hidden, fc2_weights) + fc2_biases)


    train_y = model(x, True)
    loss = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(train_y, 1e-10, 1.0)))
    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) + 
                    tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    loss += 5e-4 * regularizers
   
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        0.01,  # Base learning rate.
        global_step * BATCH_SIZE,  # Current index into the dataset.
        mnist.train.num_examples,  # Decay step.
        0.95,  # Decay rate.
        staircase=True)
    
    # Use simple momentum for the optimization.
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=global_step)
    
    # Training accuracy
    train_correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(train_y, 1))
    train_accuracy = tf.reduce_mean(tf.cast(train_correct_prediction, tf.float32))
        
    # Predictions for the test and validation, which we'll compute less often.
    eval_y = model(x, False)
    eval_correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(eval_y, 1))
    eval_accuracy = tf.reduce_mean(tf.cast(eval_correct_prediction, tf.float32))

    reshaped_test_data = numpy.reshape(mnist.test.images, [-1, 28, 28, 1])
    test_label = mnist.test.labels
    reshaped_validate_data = numpy.reshape(mnist.validation.images, [-1, 28, 28, 1])
    validate_label = mnist.validation.labels

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
            batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_x = numpy.reshape(batch_xs, [BATCH_SIZE, 28, 28, 1])
            train_feed = {x: reshaped_x, y_: batch_ys}
    
            _, step = sess.run([optimizer, global_step], feed_dict=train_feed)
            local_step += 1
            if local_step % 50 == 0:
                validate_acc = sess.run(eval_accuracy, feed_dict={x: reshaped_validate_data, y_:validate_label})
                test_acc = sess.run(eval_accuracy, feed_dict={x: reshaped_test_data, y_:test_label})
                print("Worker %d: After %d training step(s) (global step: %d), validation accuracy = %g, test accuracy = %g" %  
                  (FLAGS.worker_index, local_step, step, validate_acc, test_acc))
            if step >= FLAGS.train_steps: break
    
        time_end = time.time()
        print("Training ends @ %f" % time_end)
        training_time = time_end - time_begin
        print("Training elapsed time: %f s" % training_time)
    
        # Accuracy on test data
        print("Final test accuracy = %g" % (sess.run(eval_accuracy, feed_dict={x: reshaped_test_data, y_:test_label})))

