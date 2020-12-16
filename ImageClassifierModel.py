import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.optimizers as Optimizer
import commonConfig as cmn
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers


class ImageClassifierModel:
    @staticmethod
    def build():
        model = Sequential()
        input_shape = (cmn.__img_height__, cmn.__img_width__,3)
        model.add(layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(layers.MaxPooling2D())
        model.add(layers.Conv2D(48, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(layers.MaxPooling2D())
        model.add(layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(layers.MaxPool2D())
        model.add(layers.Dropout(rate=0.5))
        model.add(layers.Flatten())
        model.add(layers.Dense(6, activation='softmax'))
        model.compile(optimizer=Optimizer.Adam(lr=0.0001), loss='sparse_categorical_crossentropy'
                      , metrics=['accuracy'])
        return model
