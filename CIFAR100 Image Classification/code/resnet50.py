import numpy as np

from tensorflow import keras
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

class Model:
    def __init__(self):
        return None

    def predict_with_resnet50(self, image_path):
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