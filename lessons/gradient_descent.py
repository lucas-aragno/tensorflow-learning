import tensorflow as tf
import numpy as np

x = tf.placeholder("float")
y = tf.placeholder("float")
w = tf.Variable([1.0, 2.0], name="w")

# W[0] is W and W[1] is b
y_model = tf.mul(x, w[0]) + w[1]
# so this is Wx + b

# Our error function is the square of the differences
error = tf.square(y - y_model)

train_operation = tf.train.GradientDescentOptimizer(0.01).minimize(error)

model = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(model)
    for i in range(1000):
        x_value = np.random.rand()
        y_value = x_value * 2 + 6
        session.run(train_operation, feed_dict={x: x_value, y: y_value})
    w_value = session.run(w)
    print("Predicted model: {a:.3f}x + {b:.3f}".format(a=w_value[0], b=w_value[1]))
