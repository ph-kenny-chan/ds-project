
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import commonConfig as cmn


def load_train_ds():
    return tf.keras.preprocessing.image_dataset_from_directory(
        cmn.__train_dir__,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(cmn.__img_height__, cmn.__img_width__),
        batch_size=cmn.__batch_size__
    )


def load_val_ds():
    return tf.keras.preprocessing.image_dataset_from_directory(
        cmn.__train_dir__,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(cmn.__img_height__, cmn.__img_width__),
        batch_size=cmn.__batch_size__
    )

def load_test_ds():
    return tf.keras.preprocessing.image_dataset_from_directory(
        cmn.__test_dir__,
        validation_split=0.2,
        subset="test",
        seed=123,
        image_size=(cmn.__img_height__, cmn.__img_width__),
        batch_size=cmn.__batch_size__
    )
