#%%
import math
import numpy as np
import matplotlib.pyplot as plt

import os
import tensorflow as tf
from tensorflow.python.framework import ops

get_ipython().run_line_magic('matplotlib', 'inline')

# Library for Reading Images into numpy arrays through matplotlib
import matplotlib.image as mpimg
from skimage import io


#%%
def load_dataset():
    dataset = []
    # Take this hardcoded path right now
    currDir = 'C:/Users/himan/Documents/GitHub/Deep-Learning-Projects/Binary Classification/Classify-Cats/data'
    for filename in os.listdir(currDir):
        if filename.endswith('.jpg'):
            # use image reading using matplotlib image
            img = io.imread(currDir + '/' + filename)
            # Add to the list of images
            dataset.append(img)
    return dataset

#%%
dataset = load_dataset()

#%%
# Test and plot a random image from the dataset so that you know that data has been loaded correctly
imgshow = plt.imshow(dataset[0])

#%%
from skimage import data,color
from skimage.transform import resize

#%%
#Scale each image to 64 by 64
scaledDataset = []
for image in dataset:
    scaledDataset.append(resize(image, (32, 32)))

#%%
print(scaledDataset[0].shape)
imgshow = plt.imshow(scaledDataset[2])

#%%
processedData = np.asarray(scaledDataset)

#%%
# Now split this dataset into train and test set
splitRatio = (98 / 100) * len(processedData)

#%%
X_train_size = math.floor(splitRatio)
X_test_size = len(processedData) - X_train_size

#%%
X_test_size

#%%
X_train_size + X_test_size == len(processedData)

#%%
# We are just going to take first chunk of that size and remaining for testing instead of random images.
X_train = processedData[:X_train_size,:,:,:]
X_train.shape

#%%
X_test = []

#%%
#Shit logic discarded
X_test = processedData[X_train_size:X_train_size + X_test_size,:,:,:]
# Convert list to ndarray
X_test = np.asarray(X_test)

#%%
X_test.shape

#%%
imgshow = plt.imshow(X_test[199])

#%%
# Now we have our training and test set with shapes
# Time to flatten the arrays
X_train_flatten = X_train.reshape(X_train.shape[0], -1).T
X_test_flatten = X_test.reshape(X_test.shape[0], -1).T

# Since the values are already in range of 0 to 1 no need to normalize
X_train = X_train_flatten
X_test = X_test_flatten

#%%
#Create y as well here, since this is a binary classifier we will have all y=1 in training and testing set.
y_train = np.ones(X_train[1].shape)
y_test = np.ones(X_test[1].shape)

print('Number of training examples: ' + str(X_train.shape[1]))
print('Number of testing examples:' + str(X_test.shape[1]))
print('X_train shape:' + str(X_train.shape))
print('X_test shape:' + str(X_test.shape))

#%%
# Reshape y to reflect 2d shape

#y_train = np.reshape(y_train, (-1,1)).T
#y_test = np.reshape(y_test, (-1,1)).T
print('y_train shape: ' + str(y_train.shape))
print('y_test shape:' + str(y_test.shape))

#%%
# Create one hot vectors
def create_one_hot(labels, C):
    C = tf.constant(C, name='C')
    one_hot_matrix = tf.one_hot(labels, C, axis=0)
    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()

    return one_hot

#%%
Y_train = create_one_hot(y_train, 2)
Y_test = create_one_hot(y_test, 2)

print(Y_train.shape)
print(Y_test.shape)
    
#%%
def create_placeholders(n_x, n_y):
    '''Creates placeholders for tf session'''
    X = tf.placeholder(shape=(n_x,None), dtype=tf.float32, name='X')
    Y = tf.placeholder(shape=(n_y,None), dtype=tf.float32, name='Y')
    return X,Y

#%%
X,Y = create_placeholders(122,3)

#%%
def initialize_parameters():
    '''Initializes parameters or weights to be learned'''
    W1 = tf.get_variable('W1', [50,3072], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable('b1', [50,1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable('W2', [25,50], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable('b2', [25,1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable('W3', [12,25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable('b3', [12,1], initializer=tf.zeros_initializer())
    W4 = tf.get_variable('W4', [6,12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b4 = tf.get_variable('b4', [6,1], initializer=tf.zeros_initializer())
    W5 = tf.get_variable('W5', [2,6], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b5 = tf.get_variable('b5', [2,1], initializer=tf.zeros_initializer())
    
    parameters = {
        'W1':W1,
        'b1':b1,
        'W2':W2,
        'b2':b2,
        'W3':W3,
        'b3':b3,
        'W4':W4,
        'b4':b4,
        'W5':W5,
        'b5':b5
    }
    return parameters


#%%
tf.reset_default_graph()
with tf.Session() as sess:
    parameters = initialize_parameters()
    print('W1:' + str(parameters['W1']))
    sess.close()

#%%
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
    
    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    A3 = tf.nn.relu(Z3)
    Z4 = tf.add(tf.matmul(W4, A3), b4)
    A4 = tf.nn.relu(Z4)
    Z5 = tf.add(tf.matmul(W5, A4), b5)
    
    return Z5

#%%
tf.reset_default_graph()
with tf.Session() as sess:
    X,Y = create_placeholders(3072, 2)
    parameters = initialize_parameters()
    Z5 = forward_propagation(X, parameters)
    print('Z5: ' + str(Z5))


#%%
def compute_cost(Z5, Y, beta=0.05):
    '''
    This function computes the cost with regularization.

    Arguments:
    Z5 - pre activation tensor
    Y - the output tensor
    beta - regularization parameter, set to 0.01 by default
    '''
    logits = tf.transpose(Z5)
    labels = tf.transpose(Y)
    
    cost = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits = logits, multi_class_labels=labels))
    
    # Add regularization
    regularizer = tf.nn.l2_loss(Z5)
    cost += beta * regularizer
    
    return cost

#%%
tf.reset_default_graph()

with tf.Session() as sess:
    X, Y = create_placeholders(3072, 2)
    parameters = initialize_parameters()
    Z5 = forward_propagation(X, parameters)
    print(Z5.shape)
    print(Y.shape)
    cost = compute_cost(Z5, Y)
    print(Z5)
    print("cost = " + str(cost))


#%%
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
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

    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


#%%
def model(X_train, Y_train, X_test, Y_test, learning_rate, num_epochs, minibatch_size, print_cost):
    '''Implements the desired neural network'''
    ops.reset_default_graph()   # to be able to rerun the model without overwriting the tf variables
    (n_x, m) = X_train.shape
    (n_y,_) = Y_train.shape
    costs = []
    
    # Create Placeholders
    X,Y = create_placeholders(n_x, n_y)
    
    # Initialize Parameters
    parameters = initialize_parameters()
    
    # Forward Propagation
    Z5 = forward_propagation(X, parameters)
    
    # Compute Cost
    cost = compute_cost(Z5, Y)
    
    # BackPropagation: Using tensorflow Gradient Descent Optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Initialize all the variables
    init = tf.global_variables_initializer()
    
    # Start the session to run tf graph
    with tf.Session() as sess:
        # Run initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):
            epoch_cost = 0
            num_minibatches = (int)(m/minibatch_size)
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
            if epoch % 5 == True:
                costs.append(epoch_cost)
        
        #Plot the costs
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

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

    return learned_parameters
            
            


#%%
y_train.shape


#%%
parameters = model(X_train, Y_train, X_test, Y_test, 0.003, 111, 16, True)


#%%
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

    
    x = tf.placeholder("float", [3072, 1])
    Z5 = forward_propagation(x, parameters)
    p = tf.argmax(Z5)

    with tf.Session() as sess:
        prediction = sess.run(p, feed_dict = {x: X})

    return prediction


#%%
import scipy
from PIL import Image
from scipy import ndimage

my_image = 'commuter-electric-locomotive-engine-159148.jpg'

fname = "Binary Classification/images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(32,32)).reshape((1, 32*32*3)).T
my_image_prediction = predict(my_image, parameters)

plt.imshow(image)
print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))


#%%
print(X_test)


