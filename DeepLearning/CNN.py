import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
fashion_mnist = input_data.read_data_sets('data/fashion/', one_hot=True)

print("No of images in training set {}".format(fashion_mnist.train.images.shape))
print("No of labels in training set {}".format(fashion_mnist.train.labels.shape))
print("No of images in test set {}".format(fashion_mnist.test.images.shape))
print("No of labels in test set {}".format(fashion_mnist.test.labels.shape))

labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

# img1 = fashion_mnist.train.images[41].reshape(28,28)
# # Get corresponding integer label from one-hot encoded data
# label1 = np.where(fashion_mnist.train.labels[41] == 1)[0][0]
# # Plot sample
# print("y = {} ({})".format(label1, labels[label1]))
# plt.figure()
# plt.imshow(img1, cmap='Greys')
# plt.show()

X = tf.placeholder(tf.float32, [None, 784])
X_shaped = tf.reshape(X, [-1, 28, 28, 1])

y = tf.placeholder(tf.float32, [None, 10])

def conv2d(X, w):
    return tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME')

def maxpool2d(X):
    return tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

w_c1 = tf.Variable(tf.random_normal([5, 5, 1, 32]))
w_c2 = tf.Variable(tf.random_normal([5, 5, 32, 64]))
b_c1 = tf.Variable(tf.random_normal([32]))
b_c2 = tf.Variable(tf.random_normal([64]))

conv1 = tf.nn.relu(conv2d(X_shaped, w_c1) + b_c1)
conv1 = maxpool2d(conv1)

conv2 = tf.nn.relu(conv2d(conv1, w_c2) + b_c2)
conv2 = maxpool2d(conv2)

x_flattened = tf.reshape(conv2, [-1, 7 * 7 * 64])
w_fc = tf.Variable(tf.random_normal([7 * 7 * 64, 1024]))
b_fc = tf.Variable(tf.random_normal([1024]))

fc = tf.nn.relu(tf.matmul(x_flattened, w_fc) + b_fc)

w_out = tf.Variable(tf.random_normal([1024, 10]))
b_out = tf.Variable(tf.random_normal([10]))

output = tf.matmul(fc, w_out)+ b_out
yhat = tf.nn.softmax(output)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
optimiser = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(yhat, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

epochs = 10
batch_size = 100

init_op = tf.compat.v1.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    total_batch = int(len(fashion_mnist.train.labels) / batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = fashion_mnist.train.next_batch(batch_size=batch_size)
            _, c = sess.run([optimiser, cross_entropy], feed_dict={X: batch_x, y: batch_y})
            avg_cost += c / total_batch
        print('Epoch: {}, cost={}'.format(epoch+1, avg_cost))
    print(sess.run(accuracy, feed_dict={X: fashion_mnist.test.images, y: fashion_mnist.test.labels}))