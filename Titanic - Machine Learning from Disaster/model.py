# Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
import math

from tensorflow import keras

class Model:
    def __init__(self):
        self.data = []

    def create_one_hot(self,labels, C):
        C = tf.constant(C, name='C')
        one_hot_matrix = tf.one_hot(labels, C, axis=0)
        sess = tf.Session()
        one_hot = sess.run(one_hot_matrix)
        sess.close()

        return one_hot


    def create_placeholders(self,n_x, n_y):
        X = tf.placeholder(shape=(n_x, None), dtype=tf.float32, name='X')
        Y = tf.placeholder(shape=(n_y, None), dtype=tf.float32, name='Y')

        return X, Y


    def initialize_parameters(self, n_x, n_y):
        W1 = tf.get_variable(name='W1', shape=[10, n_x],
                            initializer=tf.contrib.layers.xavier_initializer(seed=1))
        b1 = tf.get_variable(name='b1', shape=[10, 1],
                            initializer=tf.zeros_initializer())
        W2 = tf.get_variable(name='W2', shape=[10, 10],
                            initializer=tf.contrib.layers.xavier_initializer(seed=1))
        b2 = tf.get_variable(name='b2', shape=[10, 1],
                            initializer=tf.zeros_initializer())
        W3 = tf.get_variable(name='W3', shape=[n_y, 10],
                            initializer=tf.contrib.layers.xavier_initializer(seed=1))
        b3 = tf.get_variable(name='b3', shape=[n_y, 1],
                            initializer=tf.zeros_initializer())

        parameters = {
            'W1': W1,
            'b1': b1,
            'W2': W2,
            'b2': b2,
            'W3': W3,
            'b3': b3
        }

        return parameters


    def forward_propagation(self, X, parameters):
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']
        W3 = parameters['W3']
        b3 = parameters['b3']

        Z1 = tf.add(tf.matmul(W1, X), b1)
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)
        A2 = tf.nn.relu(Z2)
        Z3 = tf.add(tf.matmul(W3, A2), b3)

        return Z3


    def compute_cost(self, Z, Y):
        logits = tf.transpose(Z)
        labels = tf.transpose(Y)

        cost = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits=logits,
                                                            multi_class_labels=labels))

        return cost


    def random_mini_batches(self, X, Y, mini_batch_size=64, seed=0):
        """
        Creates a list of random minibatches from (X, Y)

        Arguments:
        X -- input data, of shape (input size, number of examples)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        mini_batch_size - size of the mini-batches, integer
        seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

        Returns:
        mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
        """

        m = X.shape[1]  # number of training examples
        mini_batches = []
        np.random.seed(seed)

        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = math.floor(
            m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches


    def train_params(self, X_train, Y_train, X_test, Y_test, starter_learning_rate, num_epochs, minibatch_size, print_cost):
        '''Implements the desired neural network'''
        ops.reset_default_graph()  # to be able to rerun the model without overwriting the tf variables
        (n_x, m) = X_train.shape
        (n_y, _) = Y_train.shape
        costs = []

        # Create Placeholders
        X, Y = self.create_placeholders(n_x, n_y)

        # Initialize Parameters
        parameters = self.initialize_parameters(n_x, n_y)
        learned_parameters = parameters

        # Forward Propagation
        Z3 = self.forward_propagation(X, parameters)

        # Compute Cost
        cost = self.compute_cost(Z3, Y)

        # BackPropagation: Using tensorflow Gradient Descent Optimizer
        #global_step = tf.Variable(0, trainable=False)
        #learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                # 100000, 0.96, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=starter_learning_rate).minimize(cost)

        # Initialize all the variables
        init = tf.global_variables_initializer()

        # Start the session to run tf graph
        with tf.Session() as sess:
            # Run initialization
            sess.run(init)

            # Do the training loop
            for epoch in range(num_epochs):
                epoch_cost = 0
                num_minibatches = (int)(m / minibatch_size)
                minibatches = self.random_mini_batches(X_train, Y_train, minibatch_size, 0)

                for minibatch in minibatches:
                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch

                    # Run the session to execute the graph on a minibatch
                    _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                    epoch_cost += minibatch_cost / num_minibatches
                # Print cost for epochs
                if print_cost == True and epoch % 50 == 0:
                    print("Cost after epoch %i %f" % (epoch, epoch_cost))
                if epoch % 5 == 0:
                    costs.append(epoch_cost)
                  
                
            # Plot the costs
            plt.plot(np.squeeze(costs))
            plt.ylabel('Cost')
            plt.xlabel('Number of iterations (per tens)')
            plt.title("Learning rate=" + str(starter_learning_rate))
            plt.show()

             # Save Parameters in a varaible
            learned_parameters = sess.run(parameters)

            # Calculate correct prediction
            correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

            # Accuracy on test set
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            
            train_accuracy = accuracy.eval({X: X_train, Y: Y_train} )
            test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
            print("Train Accuracy:", train_accuracy * 100)
            print("Test Accuracy:", test_accuracy * 100)

            print("Parameters have been trained")

        return learned_parameters

    def train_with_keras_model(self, X_train, Y_train, X_test, Y_test, num_epoch, batch_size, lr=0.0001):
        m,n_x = X_train.shape
        model = keras.Sequential([
            keras.layers.Dense(input_shape=(n_x,), units=128, activation=tf.nn.relu, kernel_initializer="glorot_normal"),
            keras.layers.Dense(input_shape=(n_x,), units=64, activation=tf.nn.relu,kernel_initializer="glorot_normal"),
            keras.layers.Dense(input_shape=(n_x,), units=64, activation=tf.nn.relu,kernel_initializer="glorot_normal"),
            keras.layers.Dense(input_shape=(n_x,), units=64, activation=tf.nn.relu,kernel_initializer="glorot_normal"),
            keras.layers.Dense(input_shape=(n_x,), units=64, activation=tf.nn.relu,kernel_initializer="glorot_normal"),
            keras.layers.Dense(units=1, activation=tf.nn.softmax),
        ])

        rmsprop = keras.optimizers.RMSprop(lr=lr)
        adam = keras.optimizers.Adam(lr=lr)

        model.compile(optimizer=adam,
            loss='binary_crossentropy', 
            metrics=['accuracy'])
       
        model.fit(X_train, Y_train, epochs=num_epoch, batch_size=batch_size)

        test_loss, test_accuracy = model.evaluate(X_test, Y_test)

        print("Accuracy on the test set: " + str(test_accuracy))

        return model

    def predict(self, X, parameters):
        m,n_x = X.shape
        x = tf.placeholder("float", X.T.shape)
        Z = self.forward_propagation(x, parameters)
        print(Z)
        p = tf.argmax(Z)

        with tf.Session() as sess:
            prediction = sess.run(p, feed_dict={x: X.T})

        return prediction