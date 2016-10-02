import tensorflow as tf

# Creates a constant called X w/ a value of 35
x = tf.constant(35, name='x')
# Creates a variable called Y that has the value of X + 5
y = tf.Variable(x + 5, name='y')

# This is some tensorflow magic, here a graph is created
# With the dependencies between the variables
model = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(model)
    print(session.run(y))
