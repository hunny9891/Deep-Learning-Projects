import numpy as np
import math

from keras.models import Model
from keras.layers import Dense, AveragePooling2D, MaxPooling2D, Activation,ZeroPadding2D, Flatten, Input, Dropout, Conv2D, BatchNormalization, Add
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.initializers import glorot_uniform
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

from util import Utility

class CustomModel:
    def __init__(self):
        return None

    def identity_block(self,X, f, filters, stage, block):
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
        f1,f2,f3 = filters

        # Save the input value. This needs to be added back to the main path later.
        X_shortcut = X

        # First component of the main path
        X = Conv2D(filters= f1, kernel_size=(1,1), strides=(1,1), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        # Second component of the main path
        X = Conv2D(filters= f2, kernel_size=(f,f), strides=(1,1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        # Third component of the main path
        X = Conv2D(filters= f3, kernel_size=(1,1), strides=(1,1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

        # Add the shortcut value to the main path
        X = Add()([X_shortcut, X])
        X = Activation('relu')(X)   

        return X

    def convolutional_block(self,X, f, filters, stage, block, s=2):
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
        f1, f2, f3 = filters

        # Save input value
        X_Shortcut = X

        # First component of the main path
        X = Conv2D(f1, kernel_size=(1,1),strides=(s,s),name=conv_name_base + '2a',padding='valid',kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        # Second component of the main path
        X = Conv2D(f2, kernel_size=(f,f),strides=(1,1),name=conv_name_base + '2b',padding='same',kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        # Third component of the main path
        X = Conv2D(f3, kernel_size=(1,1),strides=(1,1),name=conv_name_base + '2c',padding='valid',kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

        # shortcut path
        X_Shortcut = Conv2D(f3, kernel_size=(1,1),strides=(s,s),name=conv_name_base + '1',padding='valid',kernel_initializer=glorot_uniform(seed=0))(X_Shortcut)
        X_Shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_Shortcut)

        # Add main and shortcut path
        X = Add()([X_Shortcut, X])
        X = Activation('relu')(X)

        return X

    def Resnet50(self,input_shape=(32,32,3), classes=10):
        """
        Implementation of the popular ResNet50 the following architecture:
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
        X = ZeroPadding2D((3,3))(X_input)

        # Stage 1
        X = Conv2D(64, (7,7), strides = (2,2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name='bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3,3), strides=(2,2))(X)
    
        # Stage 2
        X = self.convolutional_block(X, f=3, filters=[64,64,256], stage=2, block='a', s=1)
        X = self.identity_block(X, 3,[64,64,256], stage=2, block='b')
        X = self.identity_block(X,3,[64,64,256], stage=2, block='c')

        # Stage 3
        X = self.convolutional_block(X, f=3, filters=[128,128,512], stage=3, block='a', s=2)
        X = self.identity_block(X,3,[128,128,512], stage=3, block='b')
        X = self.identity_block(X,3,[128,128,512], stage=3, block='c')
        X = self.identity_block(X,3,[128,128,512], stage=3, block='d')

        # Stage 4
        X = self.convolutional_block(X, f=3, filters=[256,256,1024], stage=4, block='a', s=2)
        X = self.identity_block(X,3,[256,256,1024], stage=4, block='b')
        X = self.identity_block(X,3,[256,256,1024], stage=4, block='c')
        X = self.identity_block(X,3,[256,256,1024], stage=4, block='d')
        X = self.identity_block(X,3,[256,256,1024], stage=4, block='e')
        X = self.identity_block(X,3,[256,256,1024], stage=4, block='f')

        # Stage 5
        X = self.convolutional_block(X, f=3, filters=[512,512,2048], stage=5, block='a', s=2)
        X = self.identity_block(X,3,[512,512,2048], stage=5, block='b')
        X = self.identity_block(X,3,[512,512,2048], stage=5, block='c')

        # AVGPOOL
        X = AveragePooling2D((1,1))(X)

        # Output Layer
        X = Flatten()(X)
        X = Dense(classes, activation='softmax', name='fc'+str(classes), kernel_initializer=glorot_uniform(seed=0))(X)
        
        # Create model
        model = Model(inputs = X_input, outputs = X, name='Resnet50')
        return model

    def resnet18(self,input_shape=(32,32,3), classes=10):
        """
        Implementation of the popular ResNet50 the following architecture:
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
        X = ZeroPadding2D((3,3))(X_input)

        # Stage 1
        X = Conv2D(64, (7,7), strides = (2,2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name='bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3,3), strides=(2,2))(X)
    
        # Stage 2
        X = self.convolutional_block(X, f=3, filters=[64,64,64], stage=2, block='a', s=1)
        X = self.identity_block(X, 3,[64,64,64], stage=2, block='b')

        # Stage 3
        X = self.convolutional_block(X, f=3, filters=[64,64,128], stage=3, block='a', s=2)
        X = self.identity_block(X,3,[64,64,128], stage=3, block='b')

        # Stage 4
        X = self.convolutional_block(X, f=3, filters=[128,128,256], stage=4, block='a', s=2)
        X = self.identity_block(X,3,[128,128,256], stage=4, block='b')

        # Stage 5
        X = self.convolutional_block(X, f=3, filters=[256,256,512], stage=5, block='a', s=2)
        X = self.identity_block(X,3,[256,256,512], stage=5, block='b')

        # AVGPOOL
        X = AveragePooling2D((1,1))(X)

        # Output Layer
        X = Flatten()(X)
        X = Dense(1000, activation='relu', name='fc10000', kernel_initializer=glorot_uniform(seed=0))(X)
        X = Dense(classes, activation='softmax', name='fc'+str(classes), kernel_initializer=glorot_uniform(seed=0))(X)
        
        # Create model
        model = Model(inputs = X_input, outputs = X, name='Resnet18')
        return model



    def train(self, model,X_train, Y_train, X_test, Y_test, num_epochs, batch_size, data_augmentation=False):
        print(model.summary())
        util = Utility()
        
        # Compile the model
        model.compile(optimizer=Adam(lr=util.lr_schedule(0), epsilon=1e-8),loss='categorical_crossentropy', metrics=['accuracy'])

        modelCheckpoint = ModelCheckpoint(filepath=util.getModelPath(),monitor='val_acc',verbose=1, save_best_only=True)
        learningRateScheduler = LearningRateScheduler(util.lr_schedule)
        lrReducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0.0, patience=5, min_lr=0.5e-6)
        #earlyStop = EarlyStopping(monitor='val_loss', patience=5, min_delta=1e-3,restore_best_weights=True,mode='auto')

        # Prepare callbacks
        callbacks = [modelCheckpoint, learningRateScheduler, lrReducer]
        if not data_augmentation:
            print('Not using data augmentation.')
            # Train the model
            model.fit(X_train,Y_train,epochs=num_epochs,batch_size=batch_size,validation_data=(X_test, Y_test), shuffle=True, callbacks=callbacks)
        else:
            print('Using real-time data augmentation.')
            # This will do preprocessing and realtime data augmentation:
            datagen = ImageDataGenerator(
                # set input mean to 0 over the dataset
                featurewise_center=False,
                # set each sample mean to 0
                samplewise_center=False,
                # divide inputs by std of dataset
                featurewise_std_normalization=False,
                # divide each input by its std
                samplewise_std_normalization=False,
                # apply ZCA whitening
                zca_whitening=False,
                # epsilon for ZCA whitening
                zca_epsilon=1e-06,
                # randomly rotate images in the range (deg 0 to 180)
                rotation_range=0,
                # randomly shift images horizontally
                width_shift_range=0.1,
                # randomly shift images vertically
                height_shift_range=0.1,
                # set range for random shear
                shear_range=0.,
                # set range for random zoom
                zoom_range=0.,
                # set range for random channel shifts
                channel_shift_range=0.,
                # set mode for filling points outside the input boundaries
                fill_mode='nearest',
                # value used for fill_mode = "constant"
                cval=0.,
                # randomly flip images
                horizontal_flip=True,
                # randomly flip images
                vertical_flip=False,
                # set rescaling factor (applied before any other transformation)
                rescale=None,
                # set function that will be applied on each input
                preprocessing_function=None,
                # image data format, either "channels_first" or "channels_last"
                data_format=None,
                # fraction of images reserved for validation (strictly between 0 and 1)
                validation_split=0.0)

            # Compute quantities required for featurewise normalization
            # (std, mean, and principal components if ZCA whitening is applied).
            datagen.fit(X_train)

            # Fit the model on the batches generated by datagen.flow().
            model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                                validation_data=(X_test, Y_test),
                                epochs=num_epochs, verbose=1, workers=4,
                                callbacks=callbacks,
                                steps_per_epoch=math.ceil(len(X_train)/ batch_size))
        

        # Evaluate the model
        _,test_acc = model.evaluate(X_test, Y_test)
        print("Accuracy on the test set: " + str(test_acc * 100) + "%")

        return model
    
    def predict_with_resnet50(self, image_path, model):
        # Preprocess the image
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        print(x.shape)
        
        # Load the model 
        model = ResNet50(include_top=True, weights='imagenet')

        # Predict witht the model
        preds = model.predict(x)
        print(preds.shape)
        print('Predicted: ' + str(decode_predictions(preds, top=3)[0]))