import numpy as np
import tensorflow as tf

# Model linear regression y = Wx + b
# 1 is b/c it has one feature -> house size
x = tf.placeholder(tf.float32, [None, 1])
# The first 1 is b/c it has one Output -> house price
# The second 1 is b/c it has one Feature -> house size
W = tf.Variable(tf.zeros([1,1]))
#1 is b/c it has one Feature -> house size
b = tf.Variable(tf.zeros([1]))
product = tf.matmul(x, W)
# Prediction
y = product + b
# Actual
y_ = tf.placeholder(tf.float32, [None, 1])

# Cost function sum((y_-y)**2) or (y_ - y)^2
cost = tf.reduce_mean(tf.square(y_-y))

train_step = tf.train.GradientDescentOptimizer(0.0000001).minimize(cost)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
steps = 1000
for i in range(steps):
    # Fake data for inputs/outputs
    # xs are numbers on i
    # ys are numbers on i*2
    xs = np.array([[i]])
    ys = np.array([[2*i]])
    # Train!
    feed = { x: xs, y_: ys }
    sess.run(train_step, feed_dict=feed)
    print("After %d iteration:" % i)
    print("W: %f" % sess.run(W))
    print("b: %f" % sess.run(b))
    print("cost: %f" % sess.run(cost, feed_dict=feed))
