import librosa
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model, Sequential, layers
from tensorflow.keras.layers import (LSTM, BatchNormalization, Concatenate,
                                     Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)

path = "r.wav"

data, sr = librosa.load(path, res_type='kaiser_fast',
                        duration=2.5, sr=44100, offset=0.5)

mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)

melspec = librosa.feature.melspectrogram(data, n_mels=30)

logspec = librosa.amplitude_to_db(melspec)
#librosa.power_to_db(melspec, ref=np.max)


mfcc = np.expand_dims(mfcc, 0)
mfcc = np.expand_dims(mfcc, -1)
melspec = np.expand_dims(melspec, 0)
melspec = np.expand_dims(melspec, -1)
logspec = np.expand_dims(logspec, 0)
logspec = np.expand_dims(logspec, -1)


# def cnn_model():
#     model = Sequential([
#         Conv2D(filters=32, kernel_size=(3, 3), input_shape=(13, 216, 1),
#                activation='relu', padding='same'),
#         BatchNormalization(),
#         Dropout(rate=0.2),

#         Conv2D(filters=32, kernel_size=(3, 3),
#                activation='relu'),
#         BatchNormalization(),
#         MaxPooling2D(),
#         Dropout(rate=0.2),


#         Conv2D(filters=32, kernel_size=(3, 3),
#                activation='relu', padding='same'),
#         BatchNormalization(),
#         Dropout(rate=0.2),


#         Conv2D(filters=32, kernel_size=(3, 3),
#                activation='relu'),
#         BatchNormalization(),
#         MaxPooling2D(),
#         Dropout(rate=0.2),

#         Flatten(),

#         # Dense(64, activation='relu')
#     ])

#     return model


# model_1 = cnn_model()
# print(model_1(mfcc).shape)


def model_2():

    model_input = Input(shape=(40, 216, 1))

    # 1
    cnn_output = Conv2D(filters=32, kernel_size=(3, 3),
                        activation='relu', padding='same')(model_input)
    cnn_output = BatchNormalization()(cnn_output)
    cnn_output = Dropout(rate=0.2)(cnn_output)

    # 2
    cnn_output = Conv2D(filters=32, kernel_size=(3, 3),
                        activation='relu')(cnn_output)
    cnn_output = BatchNormalization()(cnn_output)
    cnn_output = MaxPooling2D()(cnn_output)
    cnn_output = Dropout(rate=0.2)(cnn_output)

    # 3
    cnn_output = Conv2D(filters=32, kernel_size=(3, 3),
                        activation='relu', padding='same')(cnn_output)
    cnn_output = BatchNormalization()(cnn_output)
    cnn_output = Dropout(rate=0.2)(cnn_output)

    # 4
    cnn_output = Conv2D(filters=32, kernel_size=(3, 3),
                        activation='relu')(cnn_output)
    cnn_output = BatchNormalization()(cnn_output)
    cnn_output = MaxPooling2D()(cnn_output)
    cnn_output = Dropout(rate=0.2)(cnn_output)

    # 5
    cnn_output = Conv2D(filters=32, kernel_size=(3, 3),
                        activation='relu', padding='same')(cnn_output)
    cnn_output = BatchNormalization()(cnn_output)
    cnn_output = Dropout(rate=0.2)(cnn_output)

    # 6
    cnn_output = Conv2D(filters=32, kernel_size=(3, 3),
                        activation='relu')(cnn_output)
    cnn_output = BatchNormalization()(cnn_output)
    cnn_output = MaxPooling2D()(cnn_output)
    cnn_output = Dropout(rate=0.2)(cnn_output)

    cnn_output = Flatten()(cnn_output)

    # merged = keras.layers.concatenate([cnn_outputt, lstm_output], axis=1)

    output = Dense(64, activation='relu')(cnn_output)

    output = Dense(10, activation='softmax')(output)

    model = Model(model_input, output)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # sparse_categorical_crossentropy

    return model


model = model_2()

print(model(mfcc))

