import tensorflow as tf
import time
c = tf.constant("Hello, distributed TensorFlow!")
server = tf.train.Server.create_local_server()
sess = tf.Session(server.target)  # Create a session on the server.
print server.target
print sess.run(c)

time.sleep(10)

