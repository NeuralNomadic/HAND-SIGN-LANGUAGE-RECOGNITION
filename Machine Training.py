# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 08:05:19 2024

@author: pande
"""

import zipfile
import os
import shutil
import numpy as np
from imutils import paths

def Folder_split(root_folder, list_labels ):
  for label in list_labels:
    os.makedirs(root_folder + "/train/" + label )  
    os.makedirs(root_folder + "/val/" + label ) 
    os.makedirs(root_folder + "/test/" + label ) 

    src = root_folder + "/" + label

    allPathImages = os.listdir(src)
    np.random.shuffle(allPathImages)

    train_paths, val_paths, test_paths = np.split(np.array(allPathImages), [int(len(allPathImages)*0.7) , int(len(allPathImages)*0.85)])

    train_paths = [src + "/" + name for name in train_paths.tolist()]
    val_paths = [src + "/" + name for name in val_paths.tolist()]
    test_paths = [src + "/" + name for name in test_paths.tolist()]

    for name in train_paths:
      shutil.copy(name, root_folder + "/train/" +  label )
      
    for name in val_paths:
      shutil.copy(name, root_folder + "/val/" +  label )

    for name in test_paths:
      shutil.copy(name, root_folder + "/test/" +  label )

  return root_folder + "/train" , root_folder + "/val" ,root_folder + "/test" 

root_folder = "/dataset/asl_alphabet_train/asl_alphabet_train"
image_paths = list(paths.list_images(root_folder))


list_labels =  os.listdir(root_folder)
print(len(image_paths))
print((list_labels))

import tensorflow as tf 
from tensorflow import keras
import cv2   
import random
import matplotlib.pyplot as plt  
from keras.models import Sequential
from sklearn import preprocessing
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D,Dropout,Flatten,Dense,MaxPooling2D, BatchNormalization
from sklearn.preprocessing import LabelBinarizer
from keras import layers
from tensorflow.keras.applications import ResNet50
     

aug = ImageDataGenerator(rescale=1/255.0)


train_ds = aug.flow_from_directory(train_paths, target_size=(224,224),  class_mode='categorical' ,batch_size=128,shuffle = True)
val_ds =  aug.flow_from_directory(val_paths, target_size=(224,224), class_mode='categorical', batch_size=128 )
test = aug.flow_from_directory(test_path,class_mode="categorical", target_size=(224,224), batch_size=64 )

batchX, batchy = train_ds.next()
# print(batchX[1])
print(batchy[1])
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchy.min(), batchy.max()))

baseModel = ResNet50(weights="imagenet", include_top= False, input_shape = (224,224,3))
input = baseModel.layers[0].input
output = baseModel.output
fcHead = layers.Flatten()(output)
fcHead = layers.Dense(128)(fcHead)
fcHead = layers.Activation('relu')(fcHead)
fcHead = layers.Dropout(0.5)(fcHead)

fcHead = layers.Dense(29, activation='softmax')(fcHead)
model = keras.Model(inputs=input, outputs=fcHead)


model.summary()

model.compile(loss ="categorical_crossentropy", optimizer ="adam", metrics = ["accuracy"])
# EarlyStopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)
     

H = model.fit(train_ds, validation_data = val_ds, epochs= 5)

classes = os.listdir(path_train)
classes.sort()

for i, test_image in enumerate(os.listdir(path_test)):
    image_location = path_test + test_image
    img = cv2.imread(image_location)
    img = cv2.resize(img, (224, 224))
    plt.figure()
    plt.axis('Off')
    plt.imshow(img)
    img = np.array(img) / 255.
    img = img.reshape((1, 224, 224, 3))
    prediction = np.array(model.predict(img))
    actual = test_image.split('_')[0]
    predicted = classes[prediction.argmax()]
    print('Actual class: {} \n Predicted class: {}'.format(actual, predicted))
    plt.show()
