import tensorflow as tf

# this creates a placeholder that will have 3 float values
# use None, for dinamic size
x = tf.placeholder("float", 3)
y = x * 2

with tf.Session() as session:
    # we run the Y tensor and feed dict fills the placeholder x 
    result = session.run(y, feed_dict={x: [1, 2, 3]})
    print(result)
