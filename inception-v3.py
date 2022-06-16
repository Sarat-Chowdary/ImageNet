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

train_idg = ImageDataGenerator(rotation_range=45,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               shear_range=0.2,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               fill_mode="nearest",
                               preprocessing_function=preprocess_input)

val_idg = ImageDataGenerator(preprocessing_function=preprocess_input)

# train_idg = ImageDataGenerator(vertical_flip=True,
#                                horizontal_flip=True,
#                                height_shift_range=0.1,
#                                width_shift_range=0.1,
#                                shear_range=0.2,
#                                rotation_range=90,
#                                zoom_range=[0.5, 1.0],
#                                brightness_range=[0.2, 1.0],
#                                preprocessing_function=preprocess_input)

path_1 = '../F:/datasets/imagenet-mini/train'
path_2 = '../F:/datasets/imagenet-mini/val'
start_path = '../C:/Users/Lenovo/PycharmProjects/DL/TensorFlow Directory/Projects'

relative_path_1 = os.path.relpath(path_1, start_path)
relative_path_2 = os.path.relpath(path_2, start_path)

train_gen = train_idg.flow_from_directory(
    relative_path_1,
    target_size=(224, 224),
    batch_size=32
)
val_gen = val_idg.flow_from_directory(
    relative_path_2,
    target_size=(224, 224),
    batch_size=32
)

input_shape = (224, 224, 3)
nclass = 1000

base_model = applications.inception_v3.InceptionV3(weights='imagenet',
                                                   include_top=False,
                                                   input_shape=input_shape)
base_model.trainable = False

add_model = Sequential()
add_model.add(base_model)
add_model.add(GlobalAveragePooling2D())
add_model.add(Dropout(0.5))
add_model.add(Dense(nclass,
                    activation='softmax'))

model = add_model
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.SGD(learning_rate=1e-2,
                                                 momentum=0.9),
              metrics=['accuracy'])
model.summary()
file_path = "inceptionV3.hdf5"

checkpoint = ModelCheckpoint(file_path, monitor="accuracy", verbose=1, save_best_only=True, mode='max')

early = EarlyStopping(monitor="accuracy", mode="max", patience=2)

callbacks_list = [checkpoint, early]  # early
with tf.device('/gpu:0'):
    history = model.fit(train_gen,
                        validation_data=val_gen,
                        epochs=15,
                        shuffle=True,
                        verbose=True,
                        callbacks=callbacks_list)
model.save('InceptionV5_trained/')
model.save_weights('Inception_weights_only/')

# plot the training loss and accuracy
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# FileLink(r'1.h5')
# model = keras.models.load_model('../input/image-test/1_model.h5')


# code to test out an image
# image = load_img('../input/image-test/test_botella.jpg', target_size=(224, 224))
# convert the image pixels to a numpy array
# image = img_to_array(image)
# reshape data for the model (samples, rows, columns, and channels.)
# image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
# image = preprocess_input(image)
# predict the probability across all output classes
# pred = model.predict(image)
# pred = imagenet_utils.decode_predictions(pred)
# first_pred = pred[0][0]
# print(first_pred[1], first_pred[2])

