#!/usr/bin/env python
# Copyright (C) 2016 Leapfin All Rights Reserved
"""
This script...
"""

import csv
import cv2
import numpy as np
import random

from keras import backend as K
from keras.models import Sequential
from keras.layers import Cropping2D, Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Convolution2D

import sklearn
from sklearn.model_selection import train_test_split


#image_flipped = np.fliplr(image)
#measurement_flipped = -measurement


#model.add(Lambda(lambda pix_chan: pix_chan / 255 - 0.5, input_shape=(160, 320, 3)))


samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.02)

def generator(samples, batch_size=32):
    #import pdb; pdb.set_trace()
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                #center_image = cv2.imread(name)
                center_image = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
                center_image = center_image.reshape(160, 320, 1)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 1, 80, 320  # Trimmed image format

nb_epoch = 1
dropout_rate = 0.25

model = Sequential()

# set up cropping2D layer
model.add(Cropping2D(cropping=((60, 20), (0, 0)), input_shape=(160, 320, 1)))

model.add(Lambda(lambda image: K.tf.image.resize_images(image, (66, 200)),
                 input_shape=(row, col, ch),
                 output_shape=(66, 200, ch)))
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/127.5 - 1.,
                 input_shape=(66, 200, ch),
                 output_shape=(66, 200, ch)))
####at need 66x200 from 320x160 -> 320x66
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(dropout_rate))
model.add(Dense(50))
model.add(Dropout(dropout_rate))
model.add(Dense(10))
model.add(Dropout(dropout_rate))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples),
                    nb_epoch=nb_epoch)

model.save('trained_model.h5')
