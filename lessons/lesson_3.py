import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

filename = "flower.jpg"
image = mpimg.imread(filename)
height, width, depth = image.shape
# this will print something like
# (5528, 3685, 3) => 5528 px high, 3685 px wide and 3 colors "deep"
# print(image.shape)

x = tf.Variable(image, name='x')

model = tf.initialize_all_variables()

with tf.Session() as session:
  # this swaps columns 0 and 1 and leaves 2 as it is
  # x = tf.transpose(x, perm=[1, 0, 2])
  x = tf.reverse_sequence(x, [width] * height, 1, batch_dim=0)
  session.run(model)
  result = session.run(x)

plt.imshow(image)
plt.show()
