
import numpy as np
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.models import load_model
from keras.optimizers import Adam

import scipy
import pickle
import os

from scipy import ndimage
from resnet50 import CustomModel
import matplotlib.pyplot as plt


# Constants
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

def load_data_from_keras():
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

    # Normalize the training and testing data.
    X_train = X_train / 255
    X_test = X_test / 255

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

    model_path = ROOT_DIR + '/models'
    model_file = 'custom_resnet50.h5'
    # Test whether the directory exists
    if not os.path.isdir(model_path):
        print("Directory: " + model_path + " not found. Creating directory structure.")
        os.makedirs(model_path)

    # Try loading the keras model, if not present in the directory specified then train and save from start
    # else train on loaded model.
    isModelPresent = False
    try:
        trained_model = load_model(model_path + "/" + model_file)
    except OSError as e:
        print('Model not found on the specified path.', e)
    else:
        isModelPresent = True

    if isModelPresent:
        print("Saved model has been loaded successfully.")
        trained_model.compile(optimizer=Adam(lr=0.0001, decay=1e-6, epsilon=1e-8), loss='categorical_crossentropy', metrics=['accuracy'])
        trained_model.fit(X_train, y_train, epochs=num_epochs,batch_size=64)

        test_loss, test_accuracy = trained_model.evaluate(X_test, y_test)
        print("Accuracy on the test set: " + str(test_accuracy * 100) + "%")
    else:
        print("No saved model found, creating a new model and training it.This model will be saved for further uses.")
        model = CustomModel()
        trained_model = model.train_with_custom_resnet50(X_train, y_train, X_test, y_test, num_epochs, 64, lr=0.003)

    # Save the trained model
    trained_model.save(model_path + "/" + model_file)

if __name__ == "__main__":
    main()
