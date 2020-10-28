import librosa
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model, Sequential, layers
from tensorflow.keras.layers import (LSTM, BatchNormalization, Concatenate,
                                     Conv2D, Dense, Dropout, Reshape, Flatten,
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


def model_1(number_of_outputs, n_mfcc):
    '''
    2d cnn + lstm paralel model
    '''

    model_input = Input(shape=(n_mfcc, 216, 1))

    # CNN
    # Block 1
    cnn_output = Conv2D(filters=32, kernel_size=(3, 3),
                        activation='relu', padding='same')(model_input)
    cnn_output = BatchNormalization()(cnn_output)
    cnn_output = Dropout(rate=0.2)(cnn_output)

    # Block 2
    cnn_output = Conv2D(filters=32, kernel_size=(3, 3),
                        activation='relu')(cnn_output)
    cnn_output = BatchNormalization()(cnn_output)
    cnn_output = MaxPooling2D()(cnn_output)
    cnn_output = Dropout(rate=0.2)(cnn_output)

    # Block 3
    cnn_output = Conv2D(filters=32, kernel_size=(3, 3),
                        activation='relu', padding='same')(cnn_output)
    cnn_output = BatchNormalization()(cnn_output)
    cnn_output = Dropout(rate=0.2)(cnn_output)

    # Block 4
    cnn_output = Conv2D(filters=32, kernel_size=(3, 3),
                        activation='relu')(cnn_output)
    cnn_output = BatchNormalization()(cnn_output)
    cnn_output = MaxPooling2D()(cnn_output)
    cnn_output = Dropout(rate=0.2)(cnn_output)

    # Block 5
    cnn_output = Conv2D(filters=64, kernel_size=(3, 3),
                        activation='relu', padding='same')(cnn_output)
    cnn_output = BatchNormalization()(cnn_output)
    cnn_output = Dropout(rate=0.2)(cnn_output)

    # Block 6
    cnn_output = Conv2D(filters=64, kernel_size=(3, 3),
                        activation='relu')(cnn_output)
    cnn_output = BatchNormalization()(cnn_output)
    cnn_output = MaxPooling2D()(cnn_output)
    cnn_output = Dropout(rate=0.2)(cnn_output)

    # LSTM
    lstm_input = Reshape((n_mfcc, 216))(model_input)

    lstm_output = LSTM(128, return_sequences=True)(
        lstm_input)

    lstm_output = LSTM(128, return_sequences=True)(lstm_output)

    lstm_output = LSTM(128, return_sequences=True)(lstm_output)

    # Flatten layers
    lstm_output = Flatten()(lstm_output)
    cnn_output = Flatten()(cnn_output)
    concat = Concatenate(axis=1)([cnn_output, lstm_output])

    # Dense
    output = Dense(512, activation='relu')(concat)
    output = Dense(256, activation='relu')(output)
    output = Dense(128, activation='relu')(output)

    output = Dense(number_of_outputs, activation='softmax')(output)

    model = Model(model_input, output)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


m_1 = model_1(number_of_outputs=14, n_mfcc=40)
# print(m_1.summary())
print(m_1(mfcc))


# def model_2():
# TODO: conv1D model
# model_input = Input(shape=(n_mfcc, 216, 1))
