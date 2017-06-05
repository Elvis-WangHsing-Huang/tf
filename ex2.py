import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# y = Wx+b linear model, wi

W = tf.Variable([0.3], tf.float32) # initial a= 0.3
b = tf.Variable([-0.3], tf.float32) # initial b = -0.3
x = tf.placeholder(tf.float32)
linear_model = W*x + b

# init is a handle to a TF sub-graph to initialize the globle variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# we have some predictions, but not know if they are good or not
print(sess.run(linear_model, {x:[1,2,3,4]}))

#Now, we will create a cost function
# y is the real output,
# the cost function is sum[(y - linear_model)**2]
y = tf.placeholder(tf.float32)
square_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(square_deltas)
print("Initially, W is 0.3, b is -0.3")
print("The loss is "+str(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]})))

# suppose we have a new W and b
newW = tf.assign(W,[-1.0] )
newb = tf.assign(b, [1.])
sess.run([newW, newb]) # re-assign the global variables
print("W is re-assigned as -1.0, b as 1.0")
print("The loss is now "+str(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]})))
