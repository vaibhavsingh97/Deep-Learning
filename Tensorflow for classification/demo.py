"""Supervised problem."""
import pandas as pd  # work with data as tables
import numpy as np  # use number matrics
import matplotlib.pyplot as plt
import tensorflow as tf


# step 1: Load data
dataframe = pd.read_csv('data.csv')  # dataframe objects
# remove the features we not care about
dataframe = dataframe.drop(['index', 'price', 'sq_price'], axis=1)
# we only use first 10 rows of dataset
dataframe = dataframe[0:10]
print(dataframe)

# Step 2: Adding labels
# 1 is good buy and 0 is bad buy
dataframe.loc[:, ('y1')] = [1, 1, 1, 0, 0, 1, 0, 1, 1, 1]
# y2 is a negation of y1, opposite
dataframe.loc[:, ('y2')] = dataframe['y1'] == 0
# turns True/False values to 1s and 0s
dataframe.loc[:, ('y2')] = dataframe['y2'].astype(int)

# Step 3: Prepare data for tensorflow (tensors)
# tensors are a generic version of matrix and vectors
# vector - is a list of numbers (1D tensor)
# matrix is a list of list of numbers (2D tensor)
# list of list of list of numbers (3D tensors)
# converting features in input tensor
inputX = dataframe.loc[:, ['area', 'bathrooms']].as_matrix()
# convert labels into tensors
inputY = dataframe.loc[:, ['y1', 'y2']].as_matrix()
# print(inputX)
# print(inputY)

# Step 4: Write out our hyperparameters
learning_rate = 0.000001
training__epochs = 2000
display_steps = 50
n_samples = inputY.size

# Step 5: create our compuatatin graph/Neural network
# for feature input tensor, none means any number of examples
# placeholders are gateways for data in our computational graph
x = tf.placeholder(tf.float32, [None, 2])

# create weights
# 2x2 float matrix, that We'll keep updating through our training process
# variable in tf hold and update parameters
# in memory bufers contating tensors
w = tf.Variable(tf.zeros([2, 2]))

# add biases (exampe c in y = mx + c, c is a bias)
b = tf.Variable(tf.zeros([2]))

# multiply our weights by our inputs,
# weights are how the data flows in our compuattion graph
# multiply inputs with weights and biases
y_values = tf.add(tf.matmul(x, w), b)
# apply softmax to values we just created i.e "y_values"
# softmax is our activation function
y = tf.nn.softmax(y_values)

# feeding in a matrix labels
y_ = tf.placeholder(tf.float32, [None, 2])
# perform training
# create our cost function, mean squared error
# reduced sum computes the sum of elements across dimensions of a tensor
cost = tf.reduce_sum(tf.pow(y_-y, 2))/(2*n_samples)
# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# initialize variables and tensorflow sessions
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# training loop
for i in range(training__epochs):
    sess.run(optimizer, feed_dict={x: inputX, y_: inputY}) # Take a gradient descent step using our inputs and labels

    # That's all! The rest of the cell just outputs debug messages.
    # Display logs per epoch step
    if (i) % display_steps == 0:
        cc = sess.run(cost, feed_dict={x: inputX, y_:inputY})
        print("Training step:", '%04d' % (i), "cost=", "{:.9f}".format(cc)) #, \"W=", sess.run(W), "b=", sess.run(b)

print("Optimization Finished!")
training_cost = sess.run(cost, feed_dict={x: inputX, y_: inputY})
print("Training cost=", training_cost, "W=", sess.run(w), "b=", sess.run(b), '\n')
sess.run(y, feed_dict={x: inputX })
# So It's guessing they're all good houses. That makes it get 7/10 correct
sess.run(tf.nn.softmax([1., 2.]))