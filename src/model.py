# src/model.py
# Builds the model structure (same as the notebook) so we can load or retrain it.

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

IMG_SIZE = (224, 224)


def build_model(num_classes=6, learning_rate=0.001):
    """Same architecture as the notebook: MobileNetV2 + augmentation + dropout + L2."""
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="augmentation")

    # weights=None because we load our own trained weights afterwards
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights=None)
    base_model.trainable = False

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)  # same as MobileNetV2 preprocess_input
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"])
    return model