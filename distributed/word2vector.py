import collections
import math
import random
import time
import zipfile

import numpy as np
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
FLAGS = flags.FLAGS

vocabulary_size = 50000
batch_size = 128
valid_size = 8         # Random set of words to evaluate similarity on.
valid_window = 100     # Only pick dev samples in the head of the distribution.
embedding_size = 128   # Dimension of the embedding vector.
num_sampled = 64       # Number of negative examples to sample.
train_step = 500000
skip_window = 1        # How many words to consider left and right.
num_skips = 2          # How many times to reuse an input to generate a label.


# Read data and split words.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    with zipfile.ZipFile(filename) as f:
        data = f.read(f.namelist()[0]).split()
    return data

# Build dictionary.
def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

data_index = 0
# function to generate training batch
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [ skip_window ]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

print("Loading data from worker index = %d" % FLAGS.worker_index)
words = read_data("/tmp/data/text8.zip")
print("Data size: %d" % len(words))
data, count, dictionary, reverse_dictionary = build_dataset(words)

print("Worker GRPC URL: %s" % FLAGS.worker_grpc_url)
print("Workers = %s" % FLAGS.workers)
is_chief = (FLAGS.worker_index == 0)
if is_chief: tf.reset_default_graph()

cluster = tf.train.ClusterSpec({"ps": FLAGS.parameter_servers.split(","), "worker": FLAGS.workers.split(",")})
# Construct device setter object
device_setter = tf.train.replica_device_setter(cluster=cluster)
        
# The device setter will automatically place Variables ops on separate
# parameter servers (ps). The non-Variable ops will be placed on the workers.
with tf.device(device_setter):    
    global_step = tf.Variable(0, trainable=False)
            
    # Training input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    
    # Validate input data.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    
    # Look up embeddings for inputs.
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    
    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    
    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss = tf.reduce_mean(tf.nn.nce_loss(
            nce_weights, nce_biases, embed, train_labels, num_sampled, vocabulary_size))
    
    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss, global_step=global_step)
    
    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
        
    sv = tf.train.Supervisor(is_chief=is_chief,
                             logdir="/tmp/dist-w2v",
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
        
        # Output evaluation result
        def print_eval_result():
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = "Nearest to %s:" % valid_word
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)
        
        # Perform training
        time_begin = time.time()
        average_loss = 0
        local_step = 0        
        print("Training begins @ %f" % time_begin)
        while True:
            # Training feed
            batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
            feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

            _, loss_val, step = sess.run([optimizer, loss, global_step], feed_dict=feed_dict)
            average_loss += loss_val
            if local_step % 5000 == 0:
                if local_step > 0: average_loss /= 5000
                print("Worker %d: Finished %d training steps (global step: %d): Average Loss: %g" % (FLAGS.worker_index, local_step, step, average_loss))
                average_loss = 0
        
            if local_step % 20000 == 0:
                print("Validate evaluation result at step ", local_step)
                print_eval_result()
                
            if step >= train_step: break
            local_step += 1

        time_end = time.time()
        print("Training ends @ %f" % time_end)
        training_time = time_end - time_begin
        print("Training elapsed time: %f s" % training_time)
    
        # Final word vectors.
        print("Final answer:")
        print_eval_result()
