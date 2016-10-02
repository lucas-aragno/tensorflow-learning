import tensorflow as tf
import numpy as np

data = np.random.randint(1000, size=1000)

# x = tf.constant([35, 40, 45], name='x')
# Using the Random generated data instead
x = tf.constant(data, name='x')
y = tf.Variable( x + 5, name='y')



model = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(model)
    # it calculates the same array + 5 on each component
    print(session.run(y))
