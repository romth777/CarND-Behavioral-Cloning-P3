import cv2
import numpy as np
import csv
import os
import tensorflow as tf

lines = []

with open(os.path.join(os.path.dirname(__file__), "data", "driving_log.csv")) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split("\\")[-1]
    #current_path = os.path.join(os.path.dirname(__file__), "data", filename)
    image = cv2.imread(source_path)
    #g_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)


from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model.h5')
