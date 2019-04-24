import numpy as np
from keras.applications.resnet50 import (ResNet50, decode_predictions,
                                         preprocess_input)
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, ReduceLROnPlateau)
from keras.initializers import glorot_uniform
from keras.layers import (Activation, Add, AveragePooling2D,
                          BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          Input, MaxPooling2D, ZeroPadding2D)
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

from util import Utility


class CustomModel:
    def __init__(self):
        return None

    def identity_block(self, X, f, filters, stage, block):
        """
        Implementation of the identity block as defined in Figure 3

        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network

        Returns:
        X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
        """
        # define name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # Retrieve filters
        f1, f2 = filters

        # Save the input value. This needs to be added back to the main path later.
        X_shortcut = X

        # First component of the main path
        X = Conv2D(filters=f1, kernel_size=(f, f), strides=(1, 1), padding='same',
                   name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=l2(1e-4))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        # Second component of the main path
        X = Conv2D(filters=f2, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                   name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=l2(1e-4))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)

        # Add the shortcut value to the main path
        X = Add()([X_shortcut, X])
        X = Activation('relu')(X)

        return X

    def convolutional_block(self, X, f, filters, stage, block, s=2):
        """
        Implementation of the convolutional block as defined in Figure 4

        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        s -- Integer, specifying the stride to be used

        Returns:
        X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
        """

        # Define name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # Retrieve filters
        f1, f2 = filters

        # Save input value
        X_Shortcut = X

        # First component of the main path
        X = Conv2D(f1, kernel_size=(f, f), strides=(s, s), name=conv_name_base +
                   '2a', padding='same', kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=l2(1e-4))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        # Second component of the main path
        X = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), name=conv_name_base +
                   '2b', padding='valid', kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=l2(1e-4))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)

        # shortcut path
        X_Shortcut = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), name=conv_name_base +
                            '1', padding='valid', kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=l2(1e-4))(X_Shortcut)
        X_Shortcut = BatchNormalization(
            axis=3, name=bn_name_base + '1')(X_Shortcut)

        # Add main and shortcut path
        X = Add()([X_Shortcut, X])
        X = Activation('relu')(X)

        return X

    def construct_Resnet18(self, input_shape=(32, 32, 3), classes=10):
        """
        Implementation of the popular Resnet18 the following architecture:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

        Arguments:
        input_shape -- shape of the images of the dataset
        classes -- integer, number of classes

        Returns:
        model -- a Model() instance in Keras
        """
        # Define input as a tensor of input shape
        X_input = Input(input_shape)

        # Zero Padding
        X = ZeroPadding2D((3, 3))(X_input)

        # Stage 1
        X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1',
                   kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=l2(1e-4))(X)
        X = BatchNormalization(axis=3, name='bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)

        # Stage 2
        X = self.convolutional_block(
            X, f=3, filters=[64, 64], stage=2, block='a', s=1)
        X = self.identity_block(X, 3, [64, 64], stage=2, block='b')
        # Stage 3
        X = self.convolutional_block(
            X, f=3, filters=[64, 128], stage=3, block='a', s=2)
        X = self.identity_block(X, 3, [64, 128], stage=3, block='b')
        # Stage 4
        X = self.convolutional_block(
            X, f=3, filters=[128, 256], stage=4, block='a', s=2)
        X = self.identity_block(X, 3, [128, 256], stage=4, block='b')
        # Stage 5
        X = self.convolutional_block(
            X, f=3, filters=[256, 512], stage=5, block='a', s=2)
        X = self.identity_block(X, 3, [256, 512], stage=5, block='b')

        # AVGPOOL
        X = AveragePooling2D((1, 1))(X)

        # Output Layer
        X = Flatten()(X)
        X = Dense(1000, activation='relu', name='fc10000',
                  kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=l2(1e-4))(X)
        X = Dense(classes, activation='softmax', name='fc' +
                  str(classes), kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=l2(1e-4))(X)

        # Create model
        model = Model(inputs=X_input, outputs=X, name='Resnet18')
        return model

    def construct_resnet18v2(self, input_shape=(32, 32, 3), classes=10):
        """
        Implementation of the popular Resnet18 the following architecture:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

        Arguments:
        input_shape -- shape of the images of the dataset
        classes -- integer, number of classes

        Returns:
        model -- a Model() instance in Keras
        """
        # Define input as a tensor of input shape
        X_input = Input(input_shape)
        X = X_input

        # Stage 1
        X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1',
                   kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=l2(1e-4))(X)
        X = BatchNormalization(axis=3, name='bn_conv1')(X)
        X = Activation('relu')(X)

        # Stage 2
        X = self.identity_block(X, 3, [64, 64], stage=2, block='b')
        X = self.identity_block(X, 3, [64, 64], stage=2, block='c')
        X = self.identity_block(X, 3, [64, 64], stage=2, block='d')
        # Stage 3
        X = self.convolutional_block(
            X, f=3, filters=[64, 128], stage=3, block='a', s=2)
        X = self.identity_block(X, 3, [64, 128], stage=3, block='b')
        X = self.identity_block(X, 3, [64, 128], stage=3, block='c')
        # Stage 4
        X = self.convolutional_block(
            X, f=3, filters=[128, 256], stage=4, block='a', s=2)
        X = self.identity_block(X, 3, [128, 256], stage=4, block='b')
        X = self.identity_block(X, 3, [128, 256], stage=4, block='c')
        # Stage 5
        # X = self.convolutional_block(
        #     X, f=3, filters=[256, 512], stage=5, block='a', s=2)
        # X = self.identity_block(X, 3, [256, 512], stage=5, block='b')
        # X = self.identity_block(X, 3, [256, 512], stage=5, block='c')

        # AVGPOOL
        X = AveragePooling2D((1, 1))(X)

        # Output Layer
        X = Flatten()(X)
        X = Dense(classes, activation='softmax', name='fc' +
                  str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

        # Create model
        model = Model(inputs=X_input, outputs=X, name='Resnet18')
        return model
