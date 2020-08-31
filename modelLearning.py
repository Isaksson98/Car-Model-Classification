import cv2
import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
import random
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from keras.applications.resnet50 import ResNet50

num_of_classes = 1000

dataset_path_test = 'C:/Users/Filip/Code/DeepLearning/Datasets/cars_test'
dataset_path_train = 'C:/Users/Filip/Code/DeepLearning/Datasets/cars_train'

def create_model():

    resnet_model = ResNet50(weights = 'imagenet', input_shape=(224,224,3), include_top=False)

    model = resnet_model

    model.add(Flatten())

    model.add(Dense(units=256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(units=128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(units=num_of_classes, activation='softmax'))

    adamOpti = Adam(lr = 0.0001) # Default lr = 0.0001

    model.compile(optimizer=adamOpti, loss='categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())

    return model

create_model()