import sys
import os
import tensorflow as tf

from keras.models import Sequential
from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, \
    GlobalAveragePooling2D, Concatenate
from keras import applications
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import cv2
from keras.applications.inception_v3 import preprocess_input
from imutils.video import VideoStream
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from IPython.display import FileLink
import keras
import numpy as np
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
print(tf.test.gpu_device_name())
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


val_idg = ImageDataGenerator(preprocessing_function=preprocess_input)

path_2 = '../F:/datasets/imagenet-mini/val'
start_path = '../C:/Users/Lenovo/PycharmProjects/DL/TensorFlow Directory/Projects'
relative_path_2 = os.path.relpath(path_2, start_path)


model = keras.models.load_model('inceptionV3.hdf5')

val_gen = val_idg.flow_from_directory(
    relative_path_2,
    target_size=(224, 224),
    batch_size=32
)

model.evaluate(val_gen, verbose=1)
