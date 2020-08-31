from keras.datasets import cifar10
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

import numpy as np

def normalize(x_train, x_test):

    x_train_norm = x_train.astype('float32')
    x_test_norm = x_test.astype('float32')

    x_train_norm = x_train_norm/255.0
    x_test_norm = x_test_norm/255.0

    return x_train_norm, x_test_norm

def plot_performance(history):
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    print('test')
    plt.show()
    plt.show()



# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

num_of_classes = 2

index_train = (y_train == 2).reshape(x_train.shape[0])
index_test = (y_test == 2).reshape(x_test.shape[0])

y_train_bird = index_train.astype(int)
y_test_bird = index_test.astype(int)

only_bird_x = []
only_bird_y = []

def visualize_dataset(image_list, y_bin):
    x=0
    y=0
    for j in range(len(y_bin)):
        if y_bin[j]==1 and x <5000:
            only_bird_x.append(image_list[j])
            only_bird_y.append(1)
            x=x+1
        elif y_bin[j]==0 and y <5000:
            y=y+1
            only_bird_x.append(image_list[j])
            only_bird_y.append(0)


    #fig, ax = plt.subplots(3,3)
    #for i in range(9):
    #    ax[int(i/3), i%3].axis('off')
    #    ax[int(i/3), i%3].imshow(a[i+20])
    #plt.show()

visualize_dataset(x_train, y_train_bird)
only_bird_x = np.asarray(only_bird_x)
only_bird_y = np.asarray(only_bird_y)
print(only_bird_x.shape)

only_bird_x, x_test = normalize(only_bird_x, x_test)
x_train, x_test = normalize(x_train, x_test)


# Define hyperparameters
INPUT_SIZE  = 32
BATCH_SIZE = 16
STEPS_PER_EPOCH = len(x_train)//BATCH_SIZE
EPOCHS = 15

from keras.applications.vgg16 import VGG16
from keras.models import Model

vgg16 = VGG16(include_top=False, weights='imagenet', classes=2, input_shape=(INPUT_SIZE,INPUT_SIZE,3))

# Freeze the pre-trained layers
for layer in vgg16.layers:
    layer.trainable = False

x=vgg16.output
#x=GlobalAveragePooling2D()(x)
x=Flatten(name='flatten')(x)
#x=Dense(512,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(128,activation='relu')(x) #dense layer 3
x=Dense(32,activation='relu')(x) #dense layer 3
preds=Dense(1,activation='sigmoid')(x) #final layer with softmax activation

model=Model(inputs=vgg16.input,outputs=preds)

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
print(model.summary())

aug = ImageDataGenerator(rotation_range=10, zoom_range=0.05,
	width_shift_range=0.02, height_shift_range=0.02, shear_range=0.05,
	horizontal_flip=True, fill_mode="nearest")

history = model.fit_generator(aug.flow(only_bird_x, only_bird_y, batch_size=BATCH_SIZE), validation_data=(x_test, y_test_bird), steps_per_epoch = STEPS_PER_EPOCH, epochs = EPOCHS, verbose=2)

score = model.evaluate(only_bird_x, only_bird_y, verbose=2)

for idx, metric in enumerate(model.metrics_names):
    print("{}: {}".format(metric, score[idx]))

plot_performance(history)

for i in range(9):
    #defining subplot
    plt.subplot(330+1+i)
    #plot raw pixel data
    plt.imshow(x_train[i])
#plt.show()

model.save('C:/Users/Filip/Code/DeepLearning/bird-recognition/birdRecognition.h5')
