import loadImage as load
import commonConfig as cmn
import os
import random
from matplotlib import pyplot as plt
import tensorflow as tf
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

plt.suptitle('Categories of Images')
plt.tight_layout()
plt.show()

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

# Data Visualisation - Use OS Method to read the path

# for idx, category in enumerate(cmn.__category__):
#     path = os.path.join(cmn.__train_dir__, category)
#     for img in os.listdir(path):
#         img_array = cv2.imread(os.path.join(path, img))
#         sub_plot = fig.add_subplot(3, 3, idx+1)
#         sub_plot.set_title(category)
#         plt.imshow(img_array)
#         break
#
# plt.suptitle('Categories of Images')
# plt.tight_layout()
# plt.show()

