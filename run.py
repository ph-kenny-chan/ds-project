import loadImage as load
import commonConfig as cmn
import os
import random
from matplotlib import pyplot as plt
import cv2

# load image dataset
image_ds = load.load_train_ds()

# data visualisation

fig = plt.figure(figsize=(8, 8))
for idx, category in enumerate(cmn.__category__):
    path = os.path.join(cmn.__train_dir__, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img))
        sub_plot = fig.add_subplot(3, 3, idx+1)
        sub_plot.set_title(category)
        plt.imshow(img_array)
        break

plt.suptitle('Categories of Images')
plt.tight_layout()
plt.show()

#
