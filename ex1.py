import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# the above code is to "silent" CPP logs in Tensor Flow enviroment
# 0 : show all logs
# 1 : filter out INFO
# 2 : filter out WARNINGS
# 3 : filter out ERROR logs

import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
print(node1, node2)

sess = tf.Session()
#print(sess.run([node1, node2]))
#node3 = tf.add(node1, node2)
#print("node3: ", node3)
#print("sess.run(node3): ", sess.run(node3))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b # + provides a shortcut for tf.add(a,b)
print(sess.run(adder_node, {a:3, b:4.5})) # provide a feed_dict parameters
print(sess.run(adder_node, {a:[1,3], b:[2,4]}))

add_and_triple = adder_node * 3
print(sess.run(add_and_triple, {a:3, b:4.5}))
