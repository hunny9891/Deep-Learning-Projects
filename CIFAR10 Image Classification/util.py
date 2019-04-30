import os

# Constants
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, 'models')
IMAGES_PATH = os.path.join(ROOT_DIR, 'images')
MODEL_NAME = 'custom_resnet.h5'
CSV_PATH = os.path.join(ROOT_DIR, 'trainLabels.csv')

class Utility:
    def __init__(self, *args, **kwargs):
        return None
    
    def lr_schedule(self,epoch):
        '''
        Learning Rate Schedule

        Learning Rate is scheduled according to epochs. Since the batch size is small
        learning rate decay is prompted during the training.

        Arguments:
            epoch(int): The number of epoch.
        Returns:
            lr(float): The learning rate.
        '''
        lr = 1e-3
        if epoch > 180:
            lr *= 0.5e-3
        elif epoch > 160:
            lr *= 1e-3
        elif epoch > 120:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1
        print("Learning Rate = ", lr)
        return lr

    def getModelPath(self):
            # Test whether the directory exists
        if not os.path.isdir(os.path.join(MODEL_PATH)):
            print("Directory: " + MODEL_PATH + " not found. Creating directory structure.")
            os.makedirs(MODEL_PATH)
        filepath = os.path.join(MODEL_PATH, MODEL_NAME)
        return filepath