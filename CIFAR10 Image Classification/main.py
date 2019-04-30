import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import resnet18
import resnet50
import train

from keras.datasets import cifar10
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.preprocessing import image
from scipy import ndimage
from sklearn.model_selection import train_test_split

from util import *

def load_data_from_keras():
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

    # Normalize the training and testing data.
    X_train = X_train / 255
    X_test = X_test / 255

    # Subtract pixel mean
    mean = np.mean(X_train, axis=0)
    X_train -= mean
    X_test -= mean

    return X_train, Y_train, X_test, Y_test


def load_kaggle_data():
    # load train images
    rawImagesArr = []
    trainPath = os.path.join(IMAGES_PATH, 'train')
    for imgpath in os.listdir(trainPath):
        rawImagesArr.append(image.img_to_array(image.load_img(
            os.path.join(trainPath, imgpath), target_size=(32, 32, 3))))
    
    # convert list to np array
    X_raw = np.asarray(rawImagesArr)

    # Normalize
    X_raw /= 255

    # Subtract pixel mean
    mean = np.mean(X_raw, axis=0)
    X_raw -= mean

    # load labels
    label_df = pd.read_csv(CSV_PATH)

    name_to_num = {
        'airplane': 0,
        'automobile': 1,
        'bird': 2,
        'cat': 3,
        'deer': 4,
        'dog': 5,
        'frog': 6,
        'horse': 7,
        'ship': 8,
        'truck': 9
    }

    rawLabels = []
    for name in label_df['label']:
        rawLabels.append(name_to_num[name])
    
    Y_raw = np.asarray(rawLabels)

    # Train Test split
    X_train, X_test, Y_train, Y_test = train_test_split(X_raw, Y_raw, test_size=0.16, random_state=42)

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

    print("1 for Resnet18, 2 for Resnet50 3 for Resnet18v2")
    choice = input("Choose model: ")
    choice = int(choice)
    

    #data_path = ROOT_DIR + 'dataset/cifar-100-python'
    #raw_data, meta_data, test_data = load_data(data_path)
    X_train, y_train, X_test, y_test = load_kaggle_data()
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
        model = resnet18.CustomModel().construct_Resnet18()
    elif(choice == 2):
        model = resnet50.CustomModel().construct_Rsnet50()
    elif(choice == 3):
        model = resnet18.CustomModel().construct_resnet18v2()
    
    model.summary()
    
    # Train the model
    train.train(model,X_train, y_train, X_test, y_test, num_epochs, 64, data_augmentation=True)

if __name__ == "__main__":
    main()
