import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

print("No of images in training set {}".format(mnist.train.images.shape))
print("No of labels in training set {}".format(mnist.train.labels.shape))
print("No of images in test set {}".format(mnist.test.images.shape))
print("No of labels in test set {}".format(mnist.test.labels.shape))

X = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

learning_rate = 0.1
epochs = 10
batch_size = 100

w_xh = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='w_xh')
b_h = tf.Variable(tf.random_normal([300]), name='b_h')

w_hy = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='w_hy')
b_y = tf.Variable(tf.random_normal([10]), name='b_y')

z1 = tf.add(tf.matmul(X, w_xh), b_h)
a1 = tf.nn.relu(z1)
z2 = tf.add(tf.matmul(a1, w_hy), b_y)
yhat = tf.nn.softmax(z2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(yhat), reduction_indices=[1]))

optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(yhat, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    total_bach = int(len(mnist.train.labels) / batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_bach):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            _, c = sess.run([optimiser, cross_entropy], feed_dict={X: batch_x, y: batch_y})
            avg_cost += c / total_bach
        print('Epoch: {}, cost: {}'.format(epoch+1, avg_cost))
    print(sess.run(accuracy, feed_dict={X: mnist.test.images, y: mnist.test.labels}))