import ImageClassifierModel as model
import loadImage as load
import config as cmn
import os
import random
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras as keras
import numpy as np
import cv2

# load image dataset
train_ds = load.load_train_ds()
# print(type(image_ds))
class_names = train_ds.class_names

# data visualisation - Use Tensorflow Visualise the data
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(20):
  for i in range(10):
    ax = plt.subplot(4, 4, i + 1)
    ax.set_title(class_names[labels[i]])
    ax.axis("on")
    plt.imshow(images[i].numpy().astype("uint8"))

plt.suptitle('Visualisation - Categories of Images')
plt.tight_layout()
plt.show()

print('############################')
print('########Build Model#########')
print('############################')
# Build Model
model = model.ImageClassifierModel.build()
model.summary()
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

print('############################')
print('########Train Model#########')
print('############################')
for train_img, train_lbl in train_ds:
  print('Training Image shape:')
  print(train_img.shape)
  print('Training Label shape:')
  print(train_lbl.shape)
  break

trained = model.fit(train_img, train_lbl, epochs=20, validation_split=0.30)

# Visualise Training Data
plt.figure(figsize=(10,5))
ax2 = plt.subplot(1,2,1)
ax2.plot(trained.history['accuracy'])
ax2.plot(trained.history['val_accuracy'])
ax2.set_title('Model accuracy')
ax2.set_ylabel('Accuracy')
ax2.set_xlabel('Epoch')
ax2.legend(['Train', 'Test'], loc='upper left')

ax3 = plt.subplot(1,2,2)
ax3.plot(trained.history['loss'])
ax3.plot(trained.history['val_loss'])
ax3.set_title('Model loss')
ax3.set_ylabel('Loss')
ax3.set_xlabel('Epoch')
ax3.legend(['Train', 'Test'], loc='upper left')

plt.suptitle('Visualisation - Accuracy and Loss of Training Data')
plt.tight_layout()
plt.show()
