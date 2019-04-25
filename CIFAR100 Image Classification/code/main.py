import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy
import resnet18
import resnet50
import train

from keras.datasets import cifar100
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils import to_categorical
from scipy import ndimage

from util import Utility

def load_data_from_keras():
    (X_train, Y_train), (X_test, Y_test) = cifar100.load_data(label_mode='fine')

    # Normalize the training and testing data.
    X_train = X_train / 255
    X_test = X_test / 255

    # Subtract pixel mean
    mean = np.mean(X_train, axis=0)
    X_train -= mean
    X_test -= mean

    return X_train, Y_train, X_test, Y_test

def main():

    # Make user enter the epoch size
    num_epochs = 1
    while True:
        try:
            num_epochs = input("Please enter number of epochs: ")
            if int(num_epochs) <= 0:
                print("Number should be greater than 0.")
            else:
                break
        except ValueError:
            print("You must enter a valid integer.")
    
    num_epochs = int(num_epochs)

    print("1 for Resnet18, 2 for Resnet50")
    choice = input("Choose model: ")
    choice = int(choice)
    

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

    model = None
    if(choice == 1):
        model = resnet18.CustomModel().construct_Resnet18(classes=100)
    elif(choice == 2):
        model = resnet50.CustomModel().construct_Rsnet50(classes=100)
    
    # Train the model
    train.train(model,X_train, y_train, X_test, y_test, num_epochs, 32, data_augmentation=True)

if __name__ == "__main__":
    main()
