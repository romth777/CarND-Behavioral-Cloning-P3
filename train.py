import os
import csv
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D, ELU
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

def model_just_output():
    model = Sequential()
    model.add(Flatten(input_shape=(160, 320, 3)))
    model.add(Dense(1))
    return model


def model_simple():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Flatten())
    model.add(Dense(1))
    return model


def model_like_nvidia():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(64, 64, 3)))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(MaxPooling2D((2, 2), border_mode="same"))
    model.add(BatchNormalization())

    model.add(Convolution2D(36, 5, 5, subsample=(1, 1), border_mode="valid"))
    model.add(ELU())
    model.add(Dropout(.4))
    model.add(MaxPooling2D((2, 2), border_mode="valid"))
    model.add(BatchNormalization())

    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid"))
    model.add(ELU())
    model.add(Dropout(.4))
    model.add(MaxPooling2D((2, 2), border_mode="valid"))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(1164))
    model.add(Dropout(.3))
    model.add(ELU())

    model.add(Dense(100))
    model.add(ELU())

    model.add(Dense(50))
    model.add(ELU())

    model.add(Dense(10))
    model.add(ELU())

    model.add(Dense(1))

    model.summary()
    return model

from keras.applications.vgg16 import VGG16
from keras.layers import AveragePooling2D
from keras.regularizers import l2
from keras.layers import Input
from keras.models import Model
def model_like_vgg16():
    input_image = Input(shape=(64, 64, 3))
    base_model = VGG16(input_tensor=input_image, include_top=False)

    for layer in base_model.layers[:-3]:
        layer.trainable = False

    x = base_model.get_layer("block5_conv3").output
    x = AveragePooling2D((2, 2))(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(4096, activation="elu", W_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation="elu", W_regularizer=l2(0.01))(x)
    x = Dense(2048, activation="elu", W_regularizer=l2(0.01))(x)
    x = Dense(1, activation="linear")(x)
    model = Model(inputs=base_model.input, outputs=x)
    model.summary()

    return model

def augment_brightness_camera_images(image):
    '''
    :param image: Input image
    :return: output image with reduced brightness
    '''

    # convert to HSV so that its easy to adjust brightness
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

    # randomly generate the brightness reduction factor
    # Add a constant so that it prevents the image from being completely dark
    random_bright = .25+np.random.uniform()

    # Apply the brightness reduction to the V channel
    image1[:,:,2] = image1[:,:,2]*random_bright

    # convert to RBG again
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1


def generator(samples, batch_size=32):
    num_samples = len(samples)
    use_extra_image = True
    dir_data = "data"
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                idx = 0
                if use_extra_image:
                    idx = random.randint(0, 2)

                current_path = os.path.join(os.path.dirname(__file__), dir_data, "IMG")
                image = cv2.imread(os.path.join(current_path, os.path.split(batch_sample[idx])[-1]))
                image = cv2.resize(image[55:135, :], (64, 64))
                image.astype(np.float32)

                measurement = float(batch_sample[3])

                if idx == 1:  # left
                    measurement += 0.1
                elif idx == 2:  # right
                    measurement -= 0.1

                if random.randrange(0, 2) == 1:
                    image = cv2.flip(image, 1)
                    measurement = -measurement

                images.append(image)
                angles.append(measurement)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


def train_by_generator():
    lines = []
    dir_data = "data"
    fpath = 'weights.{epoch:02d}--{loss:.2f}-{val_loss:.2f}.h5'
    cp_cb = ModelCheckpoint(filepath=fpath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')

    samples = []
    with open(os.path.join(os.path.dirname(__file__), dir_data, "driving_log.csv")) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    samples = samples[1:]
    random.shuffle(samples)

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    # model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=20, callbacks=[cp_cb, es_cb])

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    model = model_vgg_based()
    model.compile(loss='mse', optimizer='adam')

    # Preprocess incoming data, centered around zero with small standard deviation
    history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                                         validation_data=validation_generator, nb_val_samples=len(validation_samples),
                                         nb_epoch=20, verbose=1, callbacks=[cp_cb, es_cb])

    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


def train_by_once():
    lines = []
    dir_data = "data"
    fpath = 'weights.{epoch:02d}--{loss:.2f}-{val_loss:.2f}.h5'
    cp_cb = ModelCheckpoint(filepath=fpath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    es_cb = EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')
    use_extra_image = True

    with open(os.path.join(os.path.dirname(__file__), dir_data, "driving_log.csv")) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    lines = lines[1:]
    random.shuffle(lines)

    images = []
    measurements = []
    for line in lines:
        measurement = float(line[3])
        # We have very many zero steerling sample, so I will push out them from training data.
        # if measurement == 0.0 and random.randrange(0, 2) != 0:
        #    continue

        current_path = os.path.join(os.path.dirname(__file__), dir_data, "IMG")

        idx = 0
        if use_extra_image:
            idx = random.randint(0, 2)

        image = cv2.imread(os.path.join(current_path, os.path.split(line[idx])[-1]))
        image = cv2.resize(image[55:135, :], (64, 64))
        image.astype(np.float32)

        if idx == 1: # left
            measurement += 0.25
        elif idx == 2: # right
            measurement -= 0.25

        if random.randrange(0, 2) == 1:
            image = cv2.flip(image, 1)
            measurement = -measurement

        augment_brightness_camera_images(image)

        images.append(image)
        measurements.append(measurement)

    X_train = np.array(images)
    y_train = np.array(measurements)

    model = model_like_vgg16()
    #model = model_like_nvidia()
    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file='model.png', show_shapes=True)
    model.compile(loss='mse', optimizer='adam')

    history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=30, callbacks=[cp_cb, es_cb], batch_size=32)

    model.save("model.h5")

    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig("training_result.png")
    plt.show()

    plt.hist(measurements, bins=20)
    plt.savefig("input_data.png")
    plt.show()

if __name__ == '__main__':
    train_by_once()