from tensorflow import keras
from keras.layers import Conv2D, LeakyReLU, BatchNormalization
from keras.initializers import RandomNormal


def define_discriminator_model():
    layers = [
              Conv2D(32, kernel_size=(7, 7), strides=1, padding='same', activation='relu', input_shape=(128, 128, 2)),
              Conv2D(32, kernel_size=(7, 7), strides=1, padding='same', activation='relu'),
              MaxPooling2D(),
              Conv2D(64, kernel_size=(7, 7), strides=1, padding='same', activation='relu'),
              Conv2D(64, kernel_size=(5, 5), strides=1, padding='same', activation='relu'),
              MaxPooling2D(),
              Conv2D(128, kernel_size=(5, 5), strides=1, padding='same', activation='relu'),
              Conv2D(128, kernel_size=(5, 5), strides=1, padding='same', activation='relu'),
              MaxPooling2D(),
              Conv2D(256, kernel_size=(5, 5), strides=1, padding='same', activation='relu'),
              Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),
              MaxPooling2D(),
              Conv2D(512, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),
              Conv2D(512, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),
              MaxPooling2D(),
              Conv2D(512, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),
              Conv2D(512, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),
              MaxPooling2D(),
              Flatten(),
              Dense(512, activation='relu'),
              Dense(128, activation='relu'),
              Dense(16, activation='relu'),
              Dense(1, activation='sigmoid'),
    ]
    model = keras.models.Sequential(layers)
    return model