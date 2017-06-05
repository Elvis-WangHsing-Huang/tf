import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Use tf.train APIs for the first time.
import numpy as np
import tensorflow as tf

with tf.device('/cpu:0'):
    # Model parameters - initial value
    W = tf.Variable([.3], tf.float32)
    b = tf.Variable([-.3], tf.float32)

    # Model input:x and output:y
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b
    y = tf.placeholder(tf.float32)

    # Define the loss function
    loss = tf.reduce_sum(tf.square(linear_model - y))

    # Define the optimizer for the loss function
    optimizer = tf.train.GradientDescentOptimizer(0.005) # learning rate:0.01
    train = optimizer.minimize(loss)

    # provide the training data
    x_train = [1,2,3,4]
    y_train = [0,-1,-2,-3]

    # trainig loop
    init = tf.global_variables_initializer() # as defined in line 7,8
    sess = tf.Session(config=tf.ConfigProto(log_device_placement= True))
    sess.run(init) # reset values
    # train at maximum 1000 times
    for i in range(10000):
        sess.run(train, {x:x_train, y:y_train})
        #evaluate training accuracy
        curr_W, curr_b, curr_loss = sess.run([W,b, loss ], {x:x_train, y:y_train})
        #if curr_loss < 0.01:
        #    print("loss < 0.01, Done!")
        #    break

    #Finally, we print the information for the last round
    print("(%s) W:%s, b:%s, loss:%s"%(i, curr_W, curr_b, curr_loss))
    #when we choose learning-rate as 0.01, it takes 211 steps to get the good enough result (break)
    #But, if we change the learning-rate to 0.005, it takes 424 steps to break
    #However, if it is changed to 0.05, it will not conver
