# or from tensorflow import keras
# Common constant
import config as cfg

# import keras
from keras import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from keras.layers.core import Activation

# Generic Packages
import numpy as np
import os
import pandas as pd

# Machine Learning Library
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

# Plotting Libraries
import seaborn as sn;

sn.set(font_scale=1.4)
import matplotlib.pyplot as plt

# openCV
import cv2

# Tensor Flow
import tensorflow as tf

# Display Progress
from tqdm import tqdm

# Model
from ImageClassifierModel import ImageClassifierModel

# **********************************************************************************************************
class_names = cfg.__category__
class_names_label = {class_name: i for i, class_name in enumerate(class_names)}

nb_classes = len(class_names)

IMAGE_SIZE = (cfg.__img_height__, cfg.__img_width__)


# **********************************************************************************************************
# Function to Load Images & Labels
def load_data():
    datasets = [cfg.__train_dir__,
                cfg.__test_dir__]
    output = []

    # Iterate through training and test sets
    for dataset in datasets:

        images = []
        labels = []

        print("Loading {}".format(dataset))

        # Iterate through each folder corresponding to a category
        for folder in os.listdir(dataset):
            label = class_names_label[folder]

            # Iterate through each image in our folder
            for file in tqdm(os.listdir(os.path.join(dataset, folder))):
                # Get the path name of the image
                img_path = os.path.join(os.path.join(dataset, folder), file)

                # Open and resize the img
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMAGE_SIZE)

                # Append the image and its corresponding label to the output
                images.append(image)
                labels.append(label)

        images = np.array(images, dtype='float32')
        labels = np.array(labels, dtype='int32')

        output.append((images, labels))

    return output


# Loading Data (Training & Test Dataset)
(train_images, train_labels), (test_images, test_labels) = load_data()

train_images, train_labels = shuffle(train_images, train_labels, random_state=25)

# Label Dataset Shape
n_train = train_labels.shape[0]
n_test = test_labels.shape[0]

print("Number of training examples: {}".format(n_train))
print("Number of testing examples: {}".format(n_test))
print("Each image is of size: {}".format(IMAGE_SIZE))

_, train_counts = np.unique(train_labels, return_counts=True)
_, test_counts = np.unique(test_labels, return_counts=True)
pd.DataFrame({'train': train_counts, 'test': test_counts}, index=class_names).plot.bar()
plt.show()

plt.pie(train_counts,
        explode=(0, 0, 0, 0, 0, 0),
        labels=class_names,
        autopct='%1.1f%%')
plt.axis('equal')
plt.title('Proportion of each observed category')
plt.show()

# Scale the data
train_images = train_images / 255.0
test_images = test_images / 255.0


# **********************************************************************************************************
# Visualise the data [random image from training dataset]
def display_random_img(cls_names, images, labels):
    index = np.random.randint(images.shape[0])
    plt.figure()
    plt.imshow(images[index])
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title('Image #{} : '.format(index) + cls_names[labels[index]])
    plt.show()


display_random_img(class_names, train_images, train_labels)


# **********************************************************************************************************
def display_examples(cls_names, images, labels):
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("Examples of images of the dataset", fontsize=16)
    for i in range(10):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(cls_names[labels[i]])
    plt.show()


display_examples(class_names, train_images, train_labels)


# **********************************************************************************************************
# Reference: CMS

model = ImageClassifierModel.build()
print(model.summary())

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_images, train_labels, batch_size=cfg.__batch_size__, epochs=cfg.__epochs__, validation_split=cfg.__validation_split__)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\nTest accuracy:', test_acc)


# **********************************************************************************************************
def plot_accuracy_loss(history):
    """
        Plot the accuracy and the loss during the training of the nn.
    """
    fig = plt.figure(figsize=(10, 5))

    # Plot accuracy
    plt.subplot(221)
    plt.plot(history.history['accuracy'], 'bo--', label="acc")
    plt.plot(history.history['val_accuracy'], 'ro--', label="val_acc")
    plt.title("train_acc vs val_acc")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend()

    # Plot loss function
    plt.subplot(222)
    plt.plot(history.history['loss'], 'bo--', label="loss")
    plt.plot(history.history['val_loss'], 'ro--', label="val_loss")
    plt.title("train_loss vs val_loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")

    plt.legend()
    plt.show()


plot_accuracy_loss(history)

# **********************************************************************************************************
predictions = model.predict(test_images)  # Vector of probabilities
pred_labels = np.argmax(predictions, axis=1)  # We take the highest probability

display_random_img(class_names, test_images, pred_labels)
