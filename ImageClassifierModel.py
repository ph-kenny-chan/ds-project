import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.optimizers as Optimizer

import config as cmn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation


class ImageClassifierModel:
    @staticmethod
    def build():
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(cmn.__img_height__, cmn.__img_width__, 3)),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(32, (3, 3), activation='relu', padding="same"),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Flatten(),
            Dense(128, activation=tf.nn.relu),
            Dense(6),
            Activation("softmax")
        ])
        return model
