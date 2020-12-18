# Keras Tuner
import tensorflow as tf
from tensorflow import keras
from kerastuner import HyperModel, RandomSearch
from loaddata import load_data
import config as cfg
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation

from sklearn.preprocessing import LabelBinarizer

# **********************************************************************************************************
# Keras Tuner

# Loading Data (Training & Test Dataset)


(train_images, train_labels), (test_images, test_labels) = load_data()

# Scale the data
train_images = train_images / 255.0
test_images = test_images / 255.0


class_names = cfg.__category__

le = LabelBinarizer()
train_labels = le.fit_transform(train_labels)
test_labels = le.transform(test_labels)


class TestModel(HyperModel):

    def build(self, hp):
        model = keras.Sequential()
        Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(cfg.__img_height__, cfg.__img_width__, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(32, (3, 3), activation='relu', padding="same"),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Flatten(),
        Dense(units=hp.Int('units', min_value=128, max_value=256, step=32), activation=tf.nn.relu),
        Dense(6),
        Activation("softmax")
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate',
                          values=[1e-2, 1e-3, 1e-4])),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        return model


model = TestModel()

tuner = RandomSearch(
    model,
    objective='val_accuracy',
    max_trials=10,
    directory='model_dir', project_name='ImageClassifierModel')

tuner.search_space_summary()
tuner.search(train_images, train_labels,
             epochs=1,
             validation_data=(test_images, test_labels))
tuner.results_summary()
models = tuner.get_best_models(num_models=2)
