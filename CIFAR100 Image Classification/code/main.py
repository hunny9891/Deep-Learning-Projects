
import numpy as np
import scipy
from scipy import ndimage
from resnet50 import Model
from matplotlib import pyplot as plt

def main():
    my_image = 'mushrooms.jpg'

    fname = "CIFAR100 Image Classification/data/" + my_image
    image = np.array(ndimage.imread(fname, flatten=False))
    #my_image = scipy.misc.imresize(image, size=(32, 32))

    model = Model()
    model.predict_with_resnet50(fname)

    plt.imshow(image)
    
if __name__ == "__main__":
    main()