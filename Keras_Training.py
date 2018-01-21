# 1. Import necessary libraries
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
import os
import pickle
import cv2
import numpy as np
from IS_Tool2 import load_images_and_labels

# 2. Load data
X_train, y_train = load_images_and_labels('C:\\Users\\Win 8.1 VS8 X64\\Desktop\\Data2',[28,28])

# 3. Preprocess input data
X_train = X_train.astype('float32')
X_train /= 255

# 4. Define model architecture
model = Sequential()

model.add(Convolution2D(filters=32,kernel_size=(3, 3), activation='relu', padding='same', input_shape=(28, 28, 3)))
model.add(Convolution2D(filters=32,kernel_size=(3, 3), activation='relu', padding ='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax'))

# #X_train.shape
# #y_train.shape
# y_train = y_train.reshape(y_train.shape[0], 1)
y_train = np_utils.to_categorical(y_train)
#
# '''
# train_labels = np_utils.to_categorical(y_train)
# test_labels = np_utils.to_categorical(y_test)
# '''
#
# 5. Compile model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

print(model.summary())
# 6. Fit model on training data
history = model.fit(X_train, y_train,batch_size=32, nb_epoch=2, verbose=1)
#
# '''
# def display_sample(num):
#     #Print the one-hot array of this sample's label
#     print(train_labels[num])
#     #Print the label converted back to a number
#     label = train_labels[num].argmax(axis=0)
#     #Reshape the 768 value to a 28x28 image
#     image = X_train[num].argmax([28,28])
#     plt.title('Sample: %d Label: %d' %(num, label))
#     plt.imshow(image,cmap=plt.get_cmap('gray_r'))
#     plt.show()
#
# display_sample(1234)
# '''
#
model.save('D:\\model\\model_traffic_signs.h5py')

