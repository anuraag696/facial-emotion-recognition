# src/models/model.py

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

IMG_SIZE = (48, 48)
NUM_CLASSES = 7

def build_light_model():
    inputs = tf.keras.Input(shape=(48, 48, 1))

    # Block 1
    x = layers.Conv2D(32, (3,3), padding='same',
           kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2,2))(x)

    # Block 2
    x = layers.Conv2D(64, (3,3), padding='same',
            kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2,2))(x)

    # Block 3
    x = layers.Conv2D(128, (3,3), padding='same',
           kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2,2))(x)

    # Extra refinement
    x = layers.Conv2D(128, (3,3), padding='same',
           kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Replace flatten
    x = layers.GlobalAveragePooling2D()(x)

    # Head
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    return model
