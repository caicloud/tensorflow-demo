from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange
import tensorflow as tf

from tensorflow.models.image.cifar10 import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('job_name', '', 'One of "ps", "worker"')
tf.app.flags.DEFINE_string('ps_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """parameter server jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")
tf.app.flags.DEFINE_string('worker_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """worker jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")
tf.app.flags.DEFINE_integer('task_id', 0, 'Task ID of the worker/replica running the training.')


tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

tf.logging.set_verbosity(tf.logging.INFO)

def train():
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    print ('PS hosts are: %s' % ps_hosts)
    print ('Worker hosts are: %s' % worker_hosts)

    server = tf.train.Server(
        {'ps': ps_hosts, 'worker': worker_hosts},
        job_name = FLAGS.job_name,
        task_index=FLAGS.task_id)

    if FLAGS.job_name == 'ps':
        server.join()

    is_chief = (FLAGS.task_id == 0)
    if is_chief:
        if tf.gfile.Exists(FLAGS.train_dir):
            tf.gfile.DeleteRecursively(FLAGS.train_dir)
        tf.gfile.MakeDirs(FLAGS.train_dir)
  
    device_setter = tf.train.replica_device_setter(ps_tasks=1)
    with tf.device('/job:worker/task:%d' % FLAGS.task_id):
        with tf.device(device_setter):
            global_step = tf.Variable(0, trainable=False)

            # Get images and labels for CIFAR-10.
            images, labels = cifar10.distorted_inputs()

            # Build a Graph that computes the logits predictions from the
            # inference model.
            logits = cifar10.inference(images)

            # Calculate loss.
            loss = cifar10.loss(logits, labels)
            train_op = cifar10.train(loss, global_step)

            saver = tf.train.Saver()
            # We run the summaries in the same thread as the training operations by
            # passing in None for summary_op to avoid a summary_thread being started.
            # Running summaries and training operations in parallel could run out of
            # GPU memory.
            sv = tf.train.Supervisor(is_chief=is_chief,
                                     logdir=FLAGS.train_dir,
                                     init_op=tf.initialize_all_variables(),
                                     summary_op=tf.merge_all_summaries(),
                                     global_step=global_step,
                                     saver=saver,
                                     save_model_secs=60)

            tf.logging.info('%s Supervisor' % datetime.now())

            sess_config = tf.ConfigProto(allow_soft_placement=True,
                                         log_device_placement=FLAGS.log_device_placement)

            print ("Before session init")
            # Get a session.
            sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)
            print ("Session init done")

            # Start the queue runners.
            queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
            sv.start_queue_runners(sess, queue_runners)
            print ('Started %d queues for processing input data.' % len(queue_runners))
  
            """Train CIFAR-10 for a number of steps."""
            for step in xrange(FLAGS.max_steps):
                start_time = time.time()
                _, loss_value, gs = sess.run([train_op, loss, global_step])
                duration = time.time() - start_time

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if step % 10 == 0:
                    num_examples_per_step = FLAGS.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d (global_step %d), loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                    print (format_str % (datetime.now(), step, gs, loss_value, examples_per_sec, sec_per_batch))

    if is_chief:
        saver.save(sess, os.path.join(FLAGS.train_dir, 'model.ckpt'), global_step=global_step)


def main(argv=None):
    cifar10.maybe_download_and_extract()
    train()

if __name__ == '__main__':
    tf.app.run()
