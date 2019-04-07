from keras.models import load_model
from keras.preprocessing import image
import os
import numpy as np
import matplotlib.pyplot as plt

import util

#%%
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

#%%
model = load_model(os.path.join(util.MODEL_PATH, util.MODEL_NAME))

#%%
img_dir =util.IMAGES_PATH
images = []
for img_path in os.listdir(img_dir):
    img = image.load_img(img_dir + '/' + img_path, target_size=(32,32,3))
    images.append(img)

#%%
arr_images = []
for img in images:
    arr_images.append(image.img_to_array(img))



#%%
x = np.asarray(arr_images, dtype=np.float)
x = x/255
predictions = np.argmax(model.predict(x), axis=1)

#%%
for i in range(len(predictions)):
    print("Prediction is " + str(label_dict[predictions[i]]))
    plt.imshow(x[i])
    plt.show()
    _ = input("Press Enter to continue.")
    plt.close()