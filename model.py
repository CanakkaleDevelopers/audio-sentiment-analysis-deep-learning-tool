import librosa
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model, Sequential, layers
from tensorflow.keras.layers import (LSTM, BatchNormalization, Concatenate,
                                     Conv2D, Conv1D, Dense, Dropout, Reshape, Flatten,
                                     MaxPooling2D, MaxPooling1D)
from tensorflow.keras.utils import plot_model
from FeatureExtractor import *

path = "r.wav"

data, sr = librosa.load(path, res_type='kaiser_fast',
                        duration=2.5, sr=44100, offset=0.5)

mfcc_2d = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)

# melspec = librosa.feature.melspectrogram(data, n_mels=30)

# logspec = librosa.amplitude_to_db(melspec)
#librosa.power_to_db(melspec, ref=np.max)


mfcc_2d = np.expand_dims(mfcc_2d, 0)
mfcc_2d = np.expand_dims(mfcc_2d, -1)
# melspec = np.expand_dims(melspec, 0)
# melspec = np.expand_dims(melspec, -1)
# logspec = np.expand_dims(logspec, 0)
# logspec = np.expand_dims(logspec, -1)


def model_1(number_of_outputs, n_mfcc):
    '''
    2d cnn + lstm parallel model
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
# print(m_1(mfcc_2d))


# extraction 1d data

mfcc_1d = FeatureExtractor.extract("r.wav", 'mfcc')[0]
mfcc_1d = np.expand_dims(mfcc_1d, 0)
mfcc_1d = np.expand_dims(mfcc_1d, -1)


def model_2(number_of_outputs, n_mfcc):
    '''
    1d conv + lstm parallel model
    '''
    model_input = Input(shape=(n_mfcc, 1))

    # CNN
    # Block 1
    cnn_output = Conv1D(filters=32, kernel_size=(3, ),
                        activation='relu', padding='same')(model_input)
    cnn_output = BatchNormalization()(cnn_output)
    cnn_output = Dropout(rate=0.2)(cnn_output)

    # Block 2
    cnn_output = Conv1D(filters=32, kernel_size=(3, ),
                        activation='relu')(cnn_output)
    cnn_output = BatchNormalization()(cnn_output)
    cnn_output = MaxPooling1D()(cnn_output)
    cnn_output = Dropout(rate=0.2)(cnn_output)

    # Block 3
    cnn_output = Conv1D(filters=32, kernel_size=(3, ),
                        activation='relu', padding='same')(cnn_output)
    cnn_output = BatchNormalization()(cnn_output)
    cnn_output = Dropout(rate=0.2)(cnn_output)

    # Block 4
    cnn_output = Conv1D(filters=32, kernel_size=(3, ),
                        activation='relu')(cnn_output)
    cnn_output = BatchNormalization()(cnn_output)
    cnn_output = MaxPooling1D()(cnn_output)
    cnn_output = Dropout(rate=0.2)(cnn_output)

    # Block 5
    cnn_output = Conv1D(filters=64, kernel_size=(3, ),
                        activation='relu', padding='same')(cnn_output)
    cnn_output = BatchNormalization()(cnn_output)
    cnn_output = Dropout(rate=0.2)(cnn_output)

    # Block 6
    cnn_output = Conv1D(filters=64, kernel_size=(3, ),
                        activation='relu')(cnn_output)
    cnn_output = BatchNormalization()(cnn_output)
    cnn_output = MaxPooling1D()(cnn_output)
    cnn_output = Dropout(rate=0.2)(cnn_output)

    # LSTM
    lstm_output = LSTM(16, return_sequences=True)(
        model_input)

    lstm_output = LSTM(16, return_sequences=True)(lstm_output)

    lstm_output = LSTM(16, return_sequences=True)(lstm_output)

    # Flatten layers
    lstm_output = Flatten()(lstm_output)
    cnn_output = Flatten()(cnn_output)
    concat = Concatenate(axis=1)([cnn_output, lstm_output])
    print('concat :', concat.shape)
    # Dense
    output = Dense(256, activation='relu')(concat)
    output = Dense(128, activation='relu')(output)
    output = Dense(64, activation='relu')(output)

    output = Dense(number_of_outputs, activation='softmax')(output)

    model = Model(model_input, output)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


m_2 = model_2(number_of_outputs=14, n_mfcc=40)
# print(m_2(mfcc_1d))


# plot_model(m_1, to_file='2d_model_plot.png',
#            show_shapes=True, show_layer_names=True)
# plot_model(m_2, to_file='1d_model_plot.png',
#            show_shapes=True, show_layer_names=True)
