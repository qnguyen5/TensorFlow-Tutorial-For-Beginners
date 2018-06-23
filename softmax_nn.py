import tensorflow as tf
import numpy as np
from load_data import unique_labels

#BUILDING NEURAL NETWORK

#initialize placeholders
x = tf.placeholder(dtype = tf.float32, shape = [None,28,28])
y = tf.placeholder(dtype = tf.int32, shape = [None])

#flatten x
images_flat = tf.contrib.layers.flatten(x)

#fully connected layer
logits = tf.contrib.layers.fully_connected(images_flat, unique_labels.size, tf.nn.relu)
#unique_labels.size is the number of thetas, import load_data to get unique_labels.size

#defining loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels= y, logits = logits))

#define optimizer
train_op = tf.train.AdamOptimizer(learning_rate =0.01).minimize(loss)

#converting logits to label indexes
correct_pred = tf.argmax(logits, 1)

#define accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
