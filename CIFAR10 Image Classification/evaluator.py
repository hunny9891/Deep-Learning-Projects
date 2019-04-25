import os

import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import to_categorical

import util

# Load images
imgArr = []
for path in os.listdir(util.IMAGES_PATH):
    rawImg = image.load_img(os.path.join(
        util.IMAGES_PATH, path), target_size=(32, 32, 3))
    imgArr.append(image.img_to_array(rawImg))

# Convert list to np array
X = np.asarray(imgArr)

# Normalize images
X /= 255

# Subtract mean pixel
mean = np.mean(X, axis=0)
X -= mean
print(X.shape)

# Read label csv
# label_df = pd.read_csv(util.CSV_PATH)
# print(label_df.head())

label_dict = {
    0:'airplane',
    1:'automobile',
    2:'bird',
    3:'cat',
    4:'deer',
    5:'dog',
    6:'frog',
    7:'horse',
    8:'ship',
    9:'truck'
}

# label_col = label_df['label']
# lable_num = []
# for label in label_col:
#     lable_num.append(nameToNumber[label])

# Convert to np arr
# Y = np.asarray(lable_num)

# Convert to categorical
#Y = to_categorical(Y)
# print(Y.shape)

# Load model
model = load_model(os.path.join(util.MODEL_PATH, util.MODEL_NAME))
print("Model loaded")

# Evaluate
preds = np.argmax(model.predict(X), axis=1)

pred_df = pd.DataFrame(columns=['id', 'labels'])

ids = []
labels = []
i = 1
for pred in preds:
    ids.append(i)
    labels.append(label_dict[pred])
    i += 1

pred_df['id'] = ids
pred_df['labels'] = labels

pred_df.to_csv('submission.csv')
