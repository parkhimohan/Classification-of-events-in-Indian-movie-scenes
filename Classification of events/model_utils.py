from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import LSTM
from keras.layers.convolutional import Convolution2D, Convolution3D, MaxPooling2D
from keras import backend
import numpy as np
from tensorflow.python.ops import functional_ops as tf

vgg_model_16 = VGG16(include_top=True, weights="imagenet")
vgg_model_16.layers.pop()
vgg_model_16.layers.pop()
vgg_model_16.outputs = [vgg_model_16.layers[-1].output]
vgg_model_16.layers[-1].outbound_nodes = []
def get_features_batch(frames, model_name="vgg16"):
    model = vgg_model_16
    imageTensor = np.array(frames)
    modelFeature =  model.predict(imageTensor, verbose=1)
    return modelFeature

def spatial_model(number_of_classes):
    model = Sequential()
    model.add(Dense(2048, input_dim=4096, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(number_of_classes, activation='softmax'))
    return model
def lstm_model(number_of_classes=9, number_of_frames=None, input_dim=4096):
    if number_of_frames == None:
        print("Need to specify the number of frames (as timestep).")
        return
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, stateful=False, input_shape=(1, input_dim)))
    model.add(LSTM(64, return_sequences=True, stateful=False))
    model.add(LSTM(64, return_sequences=True, stateful=False))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(number_of_classes, activation='softmax'))
    return model
if __name__=="__main__":
    import cv2
    inputImage = cv2.resize(cv2.imread("testImages/test1.jpg"), (224, 224))
    from time import time
    start = time()
    vector = get_features(inputImage, 'vgg16')
    print('time taken by vgg 16:',time()-start,'seconds. Vector shape:',vector.shape)
    model = spatial_model(4)
    print(model.summary())
