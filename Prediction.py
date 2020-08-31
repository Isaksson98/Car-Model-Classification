import cv2
import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
import random
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

folder_path = 'C:/Users/Filip/Code/DeepLearning/Datasets/birdImages/'

model = keras.models.load_model('C:/Users/Filip/Code/DeepLearning/bird-recognition/birdRecognition.h5')

def prepare(filepath):

    train_datagen = ImageDataGenerator(rescale = 1./255)

    IMG_SIZE = 32
    norm_image = cv2.normalize(filepath, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    new_array = cv2.resize(norm_image, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1) 

def predicting():
    
    train_datagen = ImageDataGenerator(rescale = 1./255)

    images = []
    imageFileNames = []

    for img in os.listdir(folder_path):
        imageFileNames.append(img)
        img = os.path.join(folder_path, img)
        img = image.load_img(img, target_size=(32, 32))
        img = image.img_to_array(img)
        #img = np.expand_dims(img, axis=0)
        img = train_datagen.standardize(img)
        images.append(img)

    z = list(zip(images, imageFileNames))
    random.shuffle(z)
    images, imageFileNames = zip(*z)

    # stack up images list to pass for prediction
    images = np.asarray(images)
    #images = np.vstack(images)
    classes = model.predict(images)
    #class_probs = model.predict_proba(images)

    fig, ax = plt.subplots(3,3)
    for i in range(9):
        print(classes[i])
        final = ''
        #final += str(np.round(class_probs[i]*100,1))
        print(final)
        ax[int(i/3), i%3].grid('off')

        ax[int(i/3), i%3].set_xticks([])
        ax[int(i/3), i%3].set_yticks([])

        ax[int(i/3), i%3].set_ylabel(final, rotation=0, labelpad=20)
    
        ax[int(i/3), i%3].set_xlabel(imageFileNames[i])
        ax[int(i/3), i%3].imshow(images[i].reshape(32,32,3))    
    
    plt.show()


predicting()