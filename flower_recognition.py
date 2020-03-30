# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 21:53:56 2020

@author: newt
"""

#Building the CNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization

#Initialising the CNN
classifier=Sequential()

#Step 1 Convolution
classifier.add(Convolution2D(32,(3,3),input_shape=(128,128,3),activation='relu'))
#Step 2 Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Adding second convolution layer
classifier.add(Convolution2D(64,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Adding third convolution layer
classifier.add(Convolution2D(64,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Step 3 Flattening
classifier.add(Flatten())

#Step 4 Fully Connected layer

#for the hidden layer
classifier.add(Dense(activation='relu',units=512))
classifier.add(Dropout(0.3))
classifier.add(BatchNormalization(axis=1))
#for the second hidden layer
classifier.add(Dense(activation='relu',units=512))
classifier.add(Dropout(0.3))
classifier.add(BatchNormalization(axis=1))
#for the third hidden layer
classifier.add(Dense(activation='relu',units=512))
classifier.add(Dropout(0.3))
classifier.add(BatchNormalization(axis=1))
#for the output layer
classifier.add(Dense(activation='softmax',units=5))

#Compiling the CNN
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#Fitting the CNN to images
#The augmentation code helps in reducing overfitting
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

#Applying augmentation to the training set
training_set = train_datagen.flow_from_directory(
        'flowers/training_set',
        target_size=(128,128),
        batch_size=32,
        class_mode='categorical')

#Applying augmentation to the test set
test_set = test_datagen.flow_from_directory(
        'flowers/test_set',
        target_size=(128,128),
        batch_size=32,
        class_mode='categorical')

classifier.fit_generator(
        training_set,
        steps_per_epoch=3463,
        epochs=25,
        validation_data=test_set,
        validation_steps=860)
