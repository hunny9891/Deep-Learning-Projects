# Import libraries to be used
import math
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy
import tensorflow as tf
from scipy import ndimage
from tensorflow.python.framework import ops


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_data(data_path):
    raw_data = unpickle(data_path + '/train')
    meta_data = unpickle(data_path + '/meta')
    test_data = unpickle(data_path + '/test')
    print('Test Data Keys:' + str(test_data.keys()))

    return raw_data, meta_data, test_data


def prep_data_for_train(raw_data, test_data):
    # Prepare data to feed into the network and train.
    X_train_orig = raw_data[b'data']
    Y_train = raw_data[b'coarse_labels']
    X_test_orig = test_data[b'data']
    Y_test = test_data[b'coarse_labels']

    # Know the dimension of data
    print('Shape of training data: ' + str(X_train_orig.shape))
    print('Shape of training labels: ' + str(len(Y_train)))
    print('Shape of test data: ' + str(X_test_orig.shape))
    print('Shape of test labels: ' + str(len(Y_test)))

    # Normalize the training and testing data.
    X_train = X_train_orig / 255
    X_test = X_test_orig / 255

    print('Normalized training vector sample: ' + str(X_train[1]))
    print('Normalized testing vector sample: ' + str(X_test[1]))

    return X_train, X_test, Y_train, Y_test


def create_one_hot(labels, C):
    C = tf.constant(C, name='C')
    one_hot_matrix = tf.one_hot(labels, C, axis=0)
    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()

    return one_hot


def create_placeholders(n_x, n_y):
    X = tf.placeholder(shape=(n_x, None), dtype=tf.float32, name='X')
    Y = tf.placeholder(shape=(n_y, None), dtype=tf.float32, name='Y')

    return X, Y


def initialize_parameters():
    W1 = tf.get_variable(name='W1', shape=[1000, 3072],
                         initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable(name='b1', shape=[1000, 1],
                         initializer=tf.zeros_initializer())
    W2 = tf.get_variable(name='W2', shape=[500, 1000],
                         initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable(name='b2', shape=[500, 1],
                         initializer=tf.zeros_initializer())
    W3 = tf.get_variable(name='W3', shape=[200, 500],
                         initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable(name='b3', shape=[200, 1],
                         initializer=tf.zeros_initializer())
    W4 = tf.get_variable(name='W4', shape=[100, 200],
                         initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b4 = tf.get_variable(name='b4', shape=[100, 1],
                         initializer=tf.zeros_initializer())
    W5 = tf.get_variable(name='W5', shape=[50, 100],
                         initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b5 = tf.get_variable(name='b5', shape=[50, 1],
                         initializer=tf.zeros_initializer())
    W6 = tf.get_variable(name='W6', shape=[20, 50],
                         initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b6 = tf.get_variable(name='b6', shape=[20, 1],
                         initializer=tf.zeros_initializer())

    parameters = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2,
        'W3': W3,
        'b3': b3,
        'W4': W4,
        'b4': b4,
        'W5': W5,
        'b5': b5,
        'W6': W6,
        'b6': b6

    }

    return parameters


def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    W5 = parameters['W5']
    b5 = parameters['b5']
    W6 = parameters['W6']
    b6 = parameters['b6']

    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    A3 = tf.nn.relu(Z3)
    Z4 = tf.add(tf.matmul(W4, A3), b4)
    A4 = tf.nn.relu(Z4)
    Z5 = tf.add(tf.matmul(W5, A4), b5)
    A5 = tf.nn.relu(Z5)
    Z6 = tf.add(tf.matmul(W6, A5), b6)
    return Z6


def compute_cost(Z6, Y):
    logits = tf.transpose(Z6)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits=logits,
                                                          multi_class_labels=labels))

    return cost


def random_mini_batches(X, Y, mini_batch_size=8192*64, seed=0):
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


def model(X_train, Y_train, X_test, Y_test, starter_learning_rate, num_epochs, minibatch_size, print_cost):
    '''Implements the desired neural network'''
    ops.reset_default_graph()  # to be able to rerun the model without overwriting the tf variables
    (n_x, m) = X_train.shape
    (n_y, _) = Y_train.shape
    costs = []

    # Create Placeholders
    X, Y = create_placeholders(n_x, n_y)

    # Initialize Parameters
    parameters = initialize_parameters()

    # Forward Propagation
    Z6 = forward_propagation(X, parameters)

    # Compute Cost
    cost = compute_cost(Z6, Y)

    # BackPropagation: Using tensorflow Gradient Descent OptimizerL
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               100000, 0.96, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to run tf graph
    with tf.Session() as sess:
        # Run initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):
            print('epoch %i' % epoch)
            epoch_cost = 0
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, 0)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # Run the session to execute the graph on a minibatch
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                epoch_cost = minibatch_cost / minibatch_size
            # Print cost for epochs
            if print_cost == True and epoch % 10 == 0:
                print("Cost after epoch %i %f" % (epoch, epoch_cost))
            if epoch % 5 == 0:
                costs.append(epoch_cost)

        # Plot the costs
        plt.plot(np.squeeze(costs))
        plt.ylabel('Cost')
        plt.xlabel('Number of iterations (per tens)')
        plt.title("Learning rate=" + str(learning_rate))
        plt.show()

        # Save Parameters in a varaibles
        learned_parameters = sess.run(parameters)
        print("Parameters have been trained")

        # Calculate correct prediction
        correct_prediction = tf.equal(tf.argmax(Z6), tf.argmax(Y))

        # Accuracy on test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}) * 100)
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}) * 100)

    return learned_parameters


def predict(X, parameters):
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    W4 = tf.convert_to_tensor(parameters["W4"])
    W5 = tf.convert_to_tensor(parameters["W5"])
    b5 = tf.convert_to_tensor(parameters["b5"])
    W6 = tf.convert_to_tensor(parameters["W6"])
    b6 = tf.convert_to_tensor(parameters["b6"])

    x = tf.placeholder("float", [3072, 1])
    Z = forward_propagation(x, parameters)
    p = tf.argmax(Z)

    with tf.Session() as sess:
        prediction = sess.run(p, feed_dict={x: X})

    return prediction


def main():
    data_path = 'CIFAR100 Image Classification/dataset/cifar-100-python'
    raw_data, meta_data, test_data = load_data(data_path)
    X_train, X_test, y_train, y_test = prep_data_for_train(raw_data, test_data)
    Y_train = create_one_hot(y_train, 20)
    Y_test = create_one_hot(y_test, 20)

    print(str(Y_train.shape))
    print(str(Y_test.shape))

    learned_parameters = model(X_train.T, Y_train, X_test.T, Y_test, starter_learning_rate=0.5, num_epochs=100,
                               minibatch_size=8192, print_cost=True)

    my_image = 'mushrooms.jpg'

    fname = "D:/Deep-Learning-Projects/data/" + my_image
    image = np.array(ndimage.imread(fname, flatten=False))
    my_image = scipy.misc.imresize(image, size=(32, 32)).reshape((1, 32 * 32 * 3)).T
    my_image_prediction = predict(my_image, learned_parameters)

    plt.imshow(image)
    print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))
    print(str(meta_data[b'coarse_label_names'][np.squeeze(my_image_prediction)]))


if __name__ == "__main__":
    main()
