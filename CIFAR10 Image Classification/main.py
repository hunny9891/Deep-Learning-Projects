
import numpy as np
from tensorflow import keras
from keras.utils import to_categorical
from keras.datasets import cifar10
import scipy
import pickle

from scipy import ndimage
from resnet50 import CustomModel
import matplotlib.pyplot as plt

# Constants
ROOT_DIR = 'C:/Users/himan/Documents/GitHub/Deep-Learning-Projects/CIFAR10 Image Classification'

def load_data_from_keras():
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

    # Normalize the training and testing data.
    X_train = X_train / 255
    X_test = X_test / 255

    return X_train, Y_train, X_test, Y_test

def main():

    #data_path = ROOT_DIR + 'dataset/cifar-100-python'
    #raw_data, meta_data, test_data = load_data(data_path)
    X_train, y_train, X_test, y_test = load_data_from_keras()
    #X_train, X_test, y_train, y_test = prep_data_for_train(raw_data, test_data)
    #X_train, X_test, y_train, y_test = X_train.reshape((50000,32,32,3)), X_test.reshape((10000, 32,32,3)), y_train, y_test

    # Know the dimension of data
    print('Shape of training data: ' + str(X_train.shape))
    print('Shape of test data: ' + str(X_test.shape))
    

    # Encode y_train and y_test to one hot
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    print('Shape of training labels: ' + str(y_train.shape))
    print('Shape of test labels: ' + str(y_test.shape))

    model = CustomModel()
    trained_model = model.train_with_custom_resnet50(X_train, y_train, X_test, y_test, 200, 128)

    # Save the trained model
    trained_model.save(ROOT_DIR + '/models/custom_resnet50.h5')

    #my_image = 'mushrooms.jpg'

    #fname = "CIFAR100 Image Classification/data/" + my_image
    #image = np.array(ndimage.imread(fname, flatten=False)) 
    #my_image = scipy.misc.imresize(image, size=(32, 32))

    

    #plt.imshow(image)
    
if __name__ == "__main__":
    main()