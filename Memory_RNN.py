#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 11:28:50 2017

@author: dhingratul
"""
from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import random
# hyperparams
num_epochs = 1000
total_series_length = 100
truncated_backprop_length = 1
state_size = 4  # Number of neurons in the hidden layer
num_classes = 2  # Data is binary, 0 / 1 = Two Classes
batch_size = 5
num_batches = total_series_length//batch_size//truncated_backprop_length

# Step 1 - Data Generation
# Generate integers and corresponding binary numbers randomly selected in a
# range of 10,000. The data points are zero padded so as to make a constant
# lenght of 100


def generateData():
    x_int = random.randint(1, 10001)
    x = str(x_int)
    x = x.zfill(100)
    x = np.array(list(x))
    y = bin(x_int)[2:].zfill(100)
    y = np.array(list(y))
    x = x.reshape((batch_size, -1))
    y = y.reshape((batch_size, -1))
    return (x, y)

# Step 2 - Build the Model
batchX_placeholder = tf.placeholder(
        tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(
        tf.int32, [batch_size, truncated_backprop_length])
init_state = tf.placeholder(tf.float32, [batch_size, state_size])

# Randomly initialize weights
W = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float32)
b = tf.Variable(np.zeros((1, state_size)), dtype=tf.float32)

W2 = tf.Variable(np.random.rand(state_size, num_classes), dtype=tf.float32)
b2 = tf.Variable(np.zeros((1, num_classes)), dtype=tf.float32)
# Unpack columns
inputs_series = tf.unstack(batchX_placeholder, axis=1)
labels_series = tf.unstack(batchY_placeholder, axis=1)
# Forward pass
# State placeholder
current_state = init_state
# series of states through time
states_series = []

# For each set of inputs, forward pass through the network to get new state
# values and store all states in memory
for current_input in inputs_series:
    current_input = tf.reshape(current_input, [batch_size, 1])
    # Concatenate state and input data
    input_and_state_concatenated = tf.concat(
            axis=1, values=[current_input, current_state])
    next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)
    # Store the state in memory
    states_series.append(next_state)
    # Set current state to next one
    current_state = next_state
# Calculate loss
logits_series = [tf.matmul(state, W2) + b2 for state in states_series]
# Softmax Non-linearity
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

# Measure loss, calculate softmax again on logits, then compute cross entropy
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels) for logits,
    labels in zip(logits_series, labels_series)]
# Average Loss
total_loss = tf.reduce_mean(losses)
# Use adagrad for minimization
train_step = tf.train.AdagradOptimizer(0.2).minimize(total_loss)
# Step 3 Training the network
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss_list = []
    for epoch_idx in range(num_epochs):
        # Generate new data at every epoch
        x, y = generateData()
        # Empty hidden state
        _current_state = np.zeros((batch_size, state_size))

        print("epoch", epoch_idx)
        for batch_idx in range(num_batches):
            # layers unrolled to a limited number of time-steps:
            # truncated length
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length

            batchX = x[:, start_idx:end_idx]
            batchY = y[:, start_idx:end_idx]
            # Run the computation graph, give it the values
            _total_loss, _train_step, _current_state, _predictions_series = \
                sess.run(
                        [total_loss, train_step, current_state,
                         predictions_series],
                        feed_dict={
                                batchX_placeholder: batchX,
                                batchY_placeholder: batchY,
                                init_state: _current_state
                                })

            loss_list.append(_total_loss)

            if batch_idx % 100 == 0:
                print("Loss", _total_loss)
