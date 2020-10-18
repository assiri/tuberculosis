#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import warnings
warnings.filterwarnings('ignore')

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from functools import partial
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from imutils import paths
import cv2




INIT_LR = 1e-3
EPOCHS = 25
BS = 8


num_classes = 2


# input image dimensions
img_rows, img_cols = 96,96

print("[INFO] loading images...")
imagePaths = list(paths.list_images("./data"))
data = []
labels = []

# loop over the image paths
for imagePath in imagePaths:
    hasTuberculosis = imagePath.lower().endswith("_1.png")
    img = load_img(imagePath ,target_size=(96,96))
    x = img_to_array(img)  
    """
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (96, 96))
    """
    data.append(x)
    label = 1 if hasTuberculosis else 0
    labels.append(label)


data = np.array(data) / 255.0
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
(x_train, x_test, y_train, y_test) = train_test_split(data, labels,test_size=0.20, random_state=42)


#   Type convert and scale the test and training data
input_shape = (img_rows, img_cols, 3)

# Label Description 
label_dict = {
  0: "Normal",
  1: "Tuberculosis"
}


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()



opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy",optimizer=opt,metrics=['accuracy'])


#   Define early stopping callback
my_callbacks = [EarlyStopping(monitor='val_accuracy', patience=5, mode='max')]


hist = model.fit(x_train, y_train, batch_size=BS,steps_per_epoch=len(x_train) // BS,epochs=EPOCHS,
          #verbose=1,callbacks=my_callbacks,validation_data=(x_test, y_test),validation_steps=len(x_test) // BS,
          )

#print(hist.history.keys())


losses = pd.DataFrame(hist.history)
losses.plot()
plt.show()

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

keras_path = os.path.join(".", "keras")
os.makedirs(keras_path, exist_ok=True)

with open(os.path.join(keras_path, "model.json"), 'w') as f:
    f.write(model.to_json())
model.save_weights(os.path.join(keras_path, 'model.h5'))
model.save(os.path.join(keras_path, 'full_model.h5'))

