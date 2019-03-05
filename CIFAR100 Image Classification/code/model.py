# %%
# Import libraries to be used
import numpy as np
import tensorflow as tf
import math
from tensorflow.python.framework import ops
import pickle
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# %%
# Helper function to load train, test and metadata files.
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


rawData = unpickle(
    'C:/Users/himan/Documents/GitHub/Deep-Learning-Projects/CIFAR100 Image Classification/dataset/cifar-100-python/train')
metaData = unpickle(
    'C:/Users/himan/Documents/GitHub/Deep-Learning-Projects/CIFAR100 Image Classification/dataset/cifar-100-python/meta')
testData = unpickle(
    'C:/Users/himan/Documents/GitHub/Deep-Learning-Projects/CIFAR100 Image Classification/dataset/cifar-100-python/test')

print('Test Data Keys:' + str(testData.keys()))
# %%
imgshow = plt.imshow(rawData[b'data'][10].reshape(32, 32, 3))

# %%
# Prepare data to feed into the network and train.
X_train_orig = rawData[b'data']
Y_train_orig = rawData[b'coarse_labels']
X_test_orig = testData[b'data']
Y_test_orig = testData[b'coarse_labels']

# %%
# Know the dimension of data
print('Shape of training data: ' + str(X_train_orig.shape))
print('Shape of training labels: ' + str(len(Y_train_orig)))
print('Shape of test data: ' + str(X_test_orig.shape))
print('Shape of test labels: ' + str(len(Y_test_orig)))

# %%
# Normalize the training and testing data.
X_train = X_train_orig / 255
X_test = X_test_orig / 255

print('Normalized training vector sample: ' + str(X_train[1]))
print('Normalized testing vector sample: ' + str(X_test[1]))

# %%
'''
Below cell consists of all the helper functions to initialize the neural network model
and train it.
'''


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
    W1 = tf.get_variable(name='W1', shape=[100, 3072],
                         initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable(name='b1', shape=[100, 1],
                         initializer=tf.zeros_initializer())
    W2 = tf.get_variable(name='W2', shape=[50, 100],
                         initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable(name='b2', shape=[50, 1],
                         initializer=tf.zeros_initializer())
    W3 = tf.get_variable(name='W3', shape=[20, 50],
                         initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable(name='b3', shape=[20, 1],
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


def forward_propagation(X, parameters):
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


def compute_cost(Z3, Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits=logits,
                                                          multi_class_labels=labels))

    return cost


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
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
    Z5 = forward_propagation(X, parameters)

    # Compute Cost
    cost = compute_cost(Z5, Y)

    # BackPropagation: Using tensorflow Gradient Descent Optimizer
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               100000, 0.96, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

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
        correct_prediction = tf.equal(tf.argmax(Z5), tf.argmax(Y))

        # Accuracy on test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

    return learned_parameters


# %%
# Convert training and testing labels into one hot vector.
Y_train = create_one_hot(Y_train_orig, 20)
Y_test = create_one_hot(Y_test_orig, 20)

# %%
print(str(Y_train.shape))
print(str(Y_test.shape))

# %%
# Train the model.
learned_parameters = model(X_train.T, Y_train, X_test.T, Y_test, 0.0001, 10, 16, True)


# %%
def predict(X, parameters):
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    # W4 = tf.convert_to_tensor(parameters["W4"])
    # W5 = tf.convert_to_tensor(parameters["W5"])
    # b5 = tf.convert_to_tensor(parameters["b5"])

    x = tf.placeholder("float", [3072, 1])
    Z5 = forward_propagation(x, parameters)
    p = tf.argmax(Z5)

    with tf.Session() as sess:
        prediction = sess.run(p, feed_dict={x: X})

    return prediction


# %%
import scipy
from PIL import Image
from scipy import ndimage

my_image = '00000015_029.jpg'

fname = "Binary Classification/images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(32, 32)).reshape((1, 32 * 32 * 3)).T
my_image_prediction = predict(my_image, parameters)

plt.imshow(image)
print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))

# %%
print(str(metaData[b'coarse_label_names'][8]))
