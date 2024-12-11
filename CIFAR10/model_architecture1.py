import tensorflow as tf


def create_model_1():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(96, 3, padding="same", activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(96, 3, padding="same", activation="relu"),
        tf.keras.layers.Conv2D(96, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(3, 2),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv2D(192, 3, padding="same", activation="relu"),
        tf.keras.layers.Conv2D(192, 1, padding="valid", activation="relu"),
        tf.keras.layers.Conv2D(10, 1, padding="valid"),
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Activation("softmax")
    ])
    
    return model