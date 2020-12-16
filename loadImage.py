import tensorflow as tf
import commonConfig as cmn


def load_train_ds():
    return tf.keras.preprocessing.image_dataset_from_directory(
        cmn.__train_dir__,
        seed=123,
        image_size=(cmn.__img_height__, cmn.__img_width__),
        batch_size=cmn.__batch_size__
    )
