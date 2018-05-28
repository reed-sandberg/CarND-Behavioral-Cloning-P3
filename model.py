#!/usr/bin/env python
"""
CNN model to fit simulated autonomous driving via behavioral cloning.
"""

import csv
import random

import numpy as np

from keras import backend as K
from keras.models import Sequential
from keras.layers import Cropping2D, Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Convolution2D

import sklearn
from sklearn.model_selection import train_test_split

import cv2


# Training data shape.
TRAIN_DATA_WIDTH = 320
TRAIN_DATA_HEIGHT = 160

# Hyper-parameters.
NB_EPOCH = 1
DROPOUT_RATE = 0.25
CROP_TOP = 60
CROP_BOTTOM = 20
VALIDATION_SPLIT = 0.02
BATCH_SIZE = 32

def generator(samples, batch_size=32):
    """Generate a stream of batched samples with batch_size."""
    num_samples = len(samples)
    # Generator does not terminate, may use more than one epoch.
    while True:
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                # Read as grayscale but preserve 3-D structure with the single color channel last.
                center_image = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
                center_image = center_image.reshape(TRAIN_DATA_HEIGHT, TRAIN_DATA_WIDTH, 1)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def main():
    """Main entry point."""
    samples = []
    with open('./driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    train_samples, validation_samples = train_test_split(samples, test_size=VALIDATION_SPLIT)

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=BATCH_SIZE)
    validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

    model = Sequential()

    # Crop training images to focus on the road.
    model.add(Cropping2D(cropping=((CROP_TOP, CROP_BOTTOM),
                                   (0, 0)),
                         input_shape=(TRAIN_DATA_HEIGHT, TRAIN_DATA_WIDTH, 1)))
    # Resize training images with a built-in scaling algo.
    model.add(Lambda(lambda image: K.tf.image.resize_images(image, (66, 200)),
                     input_shape=(TRAIN_DATA_HEIGHT - CROP_TOP - CROP_BOTTOM, TRAIN_DATA_WIDTH, 1),
                     output_shape=(66, 200, 1)))
    # Normalize training data centered around zero with a small standard deviation.
    model.add(Lambda(lambda x: x/127.5 - 1.))

    # CNN model based on NVIDIA publication with 5 convolutional layers followed by 3 fully-connected layers.
    # https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    #
    # Image input 66x200 (1 channel) -> 31x98 @24 deep
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    # 14x47 @36 deep
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    # 5x22 @48 deep
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    # 3x20 @64 deep
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    # 1x18 @64 deep
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    # Fully-connected 1164 -> 100
    model.add(Dense(100))
    model.add(Dropout(DROPOUT_RATE))
    # Fully-connected 50
    model.add(Dense(50))
    model.add(Dropout(DROPOUT_RATE))
    # Fully-connected 10
    model.add(Dense(10))
    model.add(Dropout(DROPOUT_RATE))
    # Output
    model.add(Dense(1))

    # Fit the model using Adam to optimize MSE loss.
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                        validation_data=validation_generator,
                        nb_val_samples=len(validation_samples),
                        nb_epoch=NB_EPOCH)

    # Save optimized model.
    model.save('model.h5')

if __name__ == '__main__':
    main()
