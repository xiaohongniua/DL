import math
import numpy as np
import h5py
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict


# Loading the dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

# Normalize image vectors
X_train = X_train_flatten / 255
X_test = X_test_flatten / 255

# Convert training and test labels to one-hot encoding
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)


# create placeholders for data input and labels for tensorflow session

def create_placeholders(n_x, n_y):
  X = tf.placeholder(tf.float32, [n_x, None], name="X")
  Y = tf.placeholder(tf.float32, [n_y, None], name="Y")
  return X, Y


# initializing the parameters
def initialize_parameters():
  """
  number of nodes in each layer:
    12288 --> 25 --> 12 --> 6
                W1 : [25, 12288]
                b1 : [25, 1]
                W2 : [12, 25]
                b2 : [12, 1]
                W3 : [6, 12]
                b3 : [6, 1]
  """
  tf.set_random_seed(1)
  W1 = tf.get_variable("W1", [25, 12288], initializer=tf.contrib.layers.xavier_initializer(seed=1))
  b1 = tf.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())

  W2 = tf.get_variable("W2", [12, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
  b2 = tf.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())

  W3 = tf.get_variable("W3", [6, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
  b3 = tf.get_variable("b3", [6, 1], initializer=tf.zeros_initializer())

  parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
  return parameters


# forward propagation

def forward_propagation(X, parameters):
  # linear--> relu --> linear --> relu --> linear --> softmax
  # this function return last linear unit
  W1 = parameters["W1"]
  b1 = parameters["b1"]
  W2 = parameters["W2"]
  b2 = parameters["b2"]
  W3 = parameters["W3"]
  b3 = parameters["b3"]

  Z1 = tf.add(tf.matmul(W1, X), b1)
  A1 = tf.nn.relu(Z1)
  Z2 = tf.add(tf.matmul(W2, A1), b2)
  A2 = tf.nn.relu(Z2)
  Z3 = tf.add(tf.matmul(W3, A2), b3)

  return Z3

# compute cost


def compute_cost(Z3, Y):
  logits = tf.transpose(Z3)
  labels = tf.transpose(Y)
  entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
  cost = tf.reduce_mean(entropy)
  return cost


# build model
def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001, num_epochs=1500, minibatch_size=32, print_cost=True):
  ops.reset_default_graph()
  tf.set_random_seed(1)
  seed = 3
  (n_x, m) = X_train.shape
  n_y = Y_train.shape[0]
  costs = []

  X, Y = create_placeholders(n_x, n_y)
  parameters = initialize_parameters()
  Z3 = forward_propagation(X, parameters)
  cost = compute_cost(Z3, Y)
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

  init = tf.global_variables_initializer()
  with tf.Session() as sess:
    sess.run(init)

    for epoch in range(num_epochs):
      epoch_cost = 0.0
      num_minibatches = int(m / minibatch_size)
      seed = seed + 1
      minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

      for minibatch in minibatches:
        (minibatch_X, minibatch_Y) = minibatch

        _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
        epoch_cost += minibatch_cost / num_minibatches

    if print_cost == True and epoch % 100 == 0:
      print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
    if print_cost == True and epoch % 5 == 0:
      costs.append(epoch_cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    parameters = sess.run(parameters)
    correct_predictions = tf.equal(tf.argmax(Z3), tf.argmax(Y))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))

    print("Train accuracy: ", sess.run(accuracy, feed_dict={X: X_train, Y: Y_train}))
    print("Test accuracy: ", sess.run(accuracy, feed_dict={X: X_test, Y: Y_test}))

    return parameters


parameters = model(X_train, Y_train, X_test, Y_test)
