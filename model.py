import os
import csv

# open and read every line of the driving_log.csv file
#  and store on samples (list)
samples = []
with open('./car_data1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

# data generated batch-by-batch by a generator
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images, angles = [], []
            for batch_sample in batch_samples:
                # steering angle for the center image
                steering_center = float(batch_sample[3])

                # create adjusted steering measurements for the side camera images
                correction = 0.3  # Parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                
                # read in images from center, left and right cameras
                # backslash used because the data was collected on Windows OS
                filename_imgcenter = batch_sample[0].split('\\')[-1]
                filename_imgleft = batch_sample[1].split('\\')[-1]
                filename_imgright = batch_sample[2].split('\\')[-1]
                img_center = cv2.imread('./car_data1/IMG/' + filename_imgcenter)
                img_left = cv2.imread('./car_data1/IMG/' +  filename_imgleft)
                img_right = cv2.imread('./car_data1/IMG/' + filename_imgright)
                
                # Insert values at the end of images and angles lists
                images.extend([cv2.cvtColor(img_center, cv2.COLOR_BGR2RGB),
                               cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB),
                               cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)])
                angles.extend([steering_center, steering_left, steering_right])
                
            # data augmentetion: flip all the images around y-axis
            # consequently change steering angle signal (-angle)
            augmented_images, augmented_angles = [], []
            for image,angle in zip(images,angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(np.fliplr(image))
                augmented_angles.append(-angle)
                
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D

# NVIDIA Architecture
model = Sequential()

# Lambda layer normalizing and mean centered the data
model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(160,320,3)))

# Cropping the image
model.add(Cropping2D(cropping=((70,25),(0,0))))

# First convolutional layer (24 channels, 5x5 kernel)
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))

# Second convolutional layer (36 channels, 5x5 kernel)
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))

# Third convolutional layer (48 channels, 5x5 kernel)
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))

# Forth convolutional layer (64 channels, 3x3 kernel)
model.add(Convolution2D(64,3,3,activation='relu'))

# Fifth convolutional layer (64 channels, 3x3 kernel)
model.add(Convolution2D(64,3,3,activation='relu'))

# Flatten layer
model.add(Flatten())

# First fully-connected layer
model.add(Dense(100))

# Second fully-connected layer
model.add(Dense(50))

# Third fully-connected layer
model.add(Dense(10))

# Forth fully-connected layer
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*6,
                                     validation_data=validation_generator,
                                     nb_val_samples=len(validation_samples)*6, nb_epoch=5,
                                     verbose=1)

# save model
model.save('model.h5')
print('Model saved!')