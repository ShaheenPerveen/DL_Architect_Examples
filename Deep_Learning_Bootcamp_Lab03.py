## LOADING ALL THE NECESSARY LIBRARIES

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler
from keras.constraints import maxnorm
from keras.regularizers import l2
from random import shuffle

from matplotlib import pyplot as plt

import cv2
import glob
import os
import re

# DATA UPLOAD AND PREPROCESSING
positive_path = "D:/Shaheen_Data/SMILEs/positives"
negative_path = "D:/Shaheen_Data/SMILEs/negatives"


ROWS = 32
COLS = 32
CHANNELS = 1

# images = [img for img in os.listdir(train_path)]
images_pos = [img for img in os.listdir(positive_path)]
images_neg = [img for img in os.listdir(negative_path)]


print(images_pos[:10])
print(images_neg[:10])

ps = len(images_pos)
ng = len(images_neg)
tot = ps + ng

data = np.ndarray(shape=(tot,ROWS, COLS))
labels = np.ndarray(tot)

start = time.time()
for t in range(0,tot):
  if t < ps:
    for i, img_path in enumerate(images_pos):\
      img_pos = cv2.imread(os.path.join(positive_path, img_path), 0)
      img_pos = cv2.resize(img_pos, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)
    
      data[i] = img_pos
      labels[i] = 1
      
  elif t >= ps & t <= tot:
    for i, img_path in enumerate(images_neg):\
      img_neg = cv2.imread(os.path.join(negative_path, img_path), 0)
      img_neg = cv2.resize(img_neg, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)
  
      data[ps + i] = img_neg
      labels[ps + i] = 0

end = time.time()

end - start

# SHUFFLING LOADED IMAGES
ind_list = [i for i in range(tot1)]
shuffle(ind_list)
data_new  = data[ind_list,]
labels_new = labels[ind_list,]

data_new = np.array(data_new).reshape((-1, 1, ROWS, COLS)).astype('float32')

# NORMALIZING
data_new /= 255

data_new.shape

input_shape = data_new[0].shape
input_shape

# TRAIN AND TEST SPLIT
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_new, labels_new, test_size=0.3)

# CONVOLUTIONAL NEURAL NETWORKS
model = Sequential()

model.add(Convolution2D(32, (3, 3),  border_mode='valid',  activation='relu', input_shape=input_shape,  data_format='channels_first'))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

# OPTIMIZING MODEL USING DIFFERENT OPTIMIZERS
lr = 0.01 

sgd = optimizers.SGD(lr=lr, decay=1e-2, momentum=0.9, nesterov=True)

# adam = optimizers.Adam(lr=1e-3)

model.compile(loss='binary_crossentropy',
          optimizer=sgd,
          metrics=['accuracy'])

# model.compile(loss='binary_crossentropy',
#           optimizer='rmsprop',
#           metrics=['accuracy'])


## fitting the model
batch_size = 32
nb_epoch = 70

model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1)

# CALCULATING SCORES ON TEST DATA
scores = model.evaluate(x_test, y_test, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

y_pred_class = model.predict_classes(x_test)

from sklearn import metrics
metrics.accuracy_score(y_test,y_pred_class)

# CONFUSION MATRIX 
confusion_matrix_test_mlp=metrics.confusion_matrix(y_test,y_pred_class)

# CONFUSION MATRIX RELATED PARAMETERS
Accuracy_Test=(confusion_matrix_test_mlp[0,0]+confusion_matrix_test_mlp[1,1])/(confusion_matrix_test_mlp[0,0]+confusion_matrix_test_mlp[0,1]+confusion_matrix_test_mlp[1,0]+confusion_matrix_test_mlp[1,1])
TNR_Test= confusion_matrix_test_mlp[0,0]/(confusion_matrix_test_mlp[0,0] +confusion_matrix_test_mlp[0,1])
TPR_Test= confusion_matrix_test_mlp[1,1]/(confusion_matrix_test_mlp[1,0] +confusion_matrix_test_mlp[1,1])

print("Test TNR: ",TNR_Test)
print("Test TPR: ",TPR_Test)
print("Test Accuracy: ",Accuracy_Test)



# TRANSFER LEARNING
# CONVERTING GRAYSCALE IMAGES TO RGB
# storing RGB images
data2 = np.ndarray(shape=(tot, ROWS, COLS, 3))
labels2 = np.ndarray(tot)

start = time.time()

for t in range(0,tot):
  if(t % 10 == 0):
    print(t)
  if t < ps:
    for i, img_path in enumerate(images_pos1):
      img_pos1 = cv2.imread(os.path.join(positive_path, img_path), 0)
      img_pos1 = cv2.resize(img_pos1, (ROWS, COLS))
      img_pos1_rgb = np.asarray(np.dstack((img_pos1, img_pos1, img_pos1)), dtype=np.uint8)
    
      data2[i] = img_pos1_rgb
      labels2[i] = 1
      
  elif t >= ps & t <= tot:
    for i, img_path in enumerate(images_neg1):
      img_neg1 = cv2.imread(os.path.join(negative_path, img_path), 0)
      img_neg1 = cv2.resize(img_neg1, (ROWS, COLS))
      img_neg1_rgb = np.asarray(np.dstack((img_neg1, img_neg1, img_neg1)), dtype=np.uint8)
  
      data2[ps + i] = img_neg1_rgb
      labels2[ps + i] = 0


end = time.time()

end - start

ind_list = [i for i in range(tot)]
shuffle(ind_list)
data_rgb  = data2[ind_list,]
labels_rgb = labels2[ind_list,]

data_rgb = np.array(data_rgb).reshape((-1, 3, ROWS, COLS)).astype('float32')

# normalizing
data_rgb /= 255

data_rgb.shape

input_shape = data_rgb[0].shape
input_shape

from sklearn.model_selection import train_test_split
x_train_rgb, x_test_rgb, y_train_rgb, y_test_rgb = train_test_split(data_rgb, labels_rgb, test_size=0.3)

from keras.layers import Input
image_input = Input(shape=(3, 32, 32))

from keras.applications.vgg16 import VGG16

model = VGG16(input_tensor=image_input, include_top=True, weights='imagenet')

last_layer = model.get_layer('fc2').output
out = Dense(1, activation='softmax', name='output')(last_layer)

from keras.models import Model

custom_vgg_model = Model(image_input, out)
custom_vgg_model.summary()

for layer in custom_vgg_model.layers[:-1]:
    layer.trainable = False


custom_vgg_model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])


hist = custom_vgg_model.fit(x_train_rgb, y_train_rgb, batch_size=32, epochs=nb_epoch, verbose=1, validation_data=(x_test_rgb, y_test_rgb))

(loss, accuracy) = custom_vgg_model.evaluate(x_test_rgb, y_test_rgb, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))


# TRANSFER LEARNING WITH FINAL LAYER AS MLP
model = VGG16(input_tensor=image_input, include_top=True, weights='imagenet')

last_layer = model.get_layer('block5_pool').output
x = Flatten(name='flatten')(last_layer)
x = Dense(128, activation='relu', name='fc1')(x)
x = Dense(128, activation='relu', name='fc2')(x)
out = Dense(1, activation='softmax', name='output')(x)
custom_vgg_model2 = Model(image_input, out)

# freeze all the layers except the dense layers
for layer in custom_vgg_model2.layers[:-3]:
    layer.trainable = False

hist = custom_vgg_model2.fit(x_train_rgb, y_train_rgb, batch_size=32, epochs=nb_epoch, verbose=1, validation_data=(x_test_rgb, y_test_rgb))

(loss, accuracy) = custom_vgg_model2.evaluate(x_test_rgb, y_test_rgb, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))

