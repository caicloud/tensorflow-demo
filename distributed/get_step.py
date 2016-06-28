import sys
import time
import tensorflow as tf


is_chief = True
tf.reset_default_graph()

cluster = tf.train.ClusterSpec({"ps": "tf-ps0:2222,tf-ps1:2222".split(","), "worker": "180.101.191.78:30001,180.101.191.78:30002,180.101.191.78:30003".split(",")})
# Construct device setter object
device_setter = tf.train.replica_device_setter(cluster=cluster)
        
# The device setter will automatically place Variables ops on separate
# parameter servers (ps). The non-Variable ops will be placed on the workers.
with tf.device(device_setter):
    a =  tf.Variable(0, trainable=False)

    sv = tf.train.Supervisor(is_chief=is_chief,
                             init_op=tf.initialize_all_variables(),
                             recovery_wait_secs=1)
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True,
                                 device_filters=["/job:ps", "/job:worker/task:0"])

               
    with sv.prepare_or_wait_for_session("grpc://180.101.191.78:30001", config=sess_config) as sess:       
	print sess.run(a)

        
