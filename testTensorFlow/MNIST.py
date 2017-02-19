#!/usr/bin/python

#Load matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#Loading the mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#Import tensor flow
import tensorflow as tf
sess = tf.InteractiveSession()

#Allocating sizes for the images
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#Defining the convolution
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#Defining the max-pooling
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

### LAYER 1 ###

#Defining the first convolutional layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

#Resizing the pictures
x_image = tf.reshape(x, [-1,28,28,1])

#Defining the operations in the first layer
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

### LAYER 2 ###

#Defining the second convolutioinal layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

#Defining the operations in the second layer
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


### LAYER 3 ###

#Defining the third layer which is fully connected
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

#Defining the operations in the third layer
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


#Performing dropout
keep_prob = tf.placeholder(tf.float32) #Proba to keep a neuron's output
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

### LAYER 4 ###

#Defining the fourth layer which is just a soft max
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

#Defining the operations in the fourth layer
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#Using cross-entropy as a loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))

#Defining the learning rate
learningRate = 1e-4

#Training
train_step = tf.train.AdamOptimizer(learningRate).minimize(cross_entropy)

#Determining the number of correct predictions
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

#Averaging the number of correct predictions
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Performing the initialization in the back-end
sess.run(tf.global_variables_initializer())

#Doing 1000 training steps
iterations = []
for i in range(2000):
  batch = mnist.train.next_batch(50)
  train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
  print("step %d, training accuracy %g"%(i, train_accuracy))
  _, loss_val =  sess.run([train_step,cross_entropy], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
  iterations.append(loss_val)

plt.plot(iterations)
plt.show()

#Number of correct prediction
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


