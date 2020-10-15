import librosa
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, Sequential, Input, Model
from tensorflow.keras.layers import Concatenate, Dense, Conv2D, Flatten, MaxPooling2D, LSTM, Dropout, BatchNormalization

path = "r.wav"

data, sr = librosa.load(path, res_type='kaiser_fast',
                        duration=2.5, sr=44100, offset=0.5)

mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=30)
melspec = librosa.feature.melspectrogram(data, n_mels=30)
logspec = librosa.amplitude_to_db(melspec)
#librosa.power_to_db(melspec, ref=np.max)


mfcc = np.expand_dims(mfcc, 0)
mfcc = np.expand_dims(mfcc, -1)
melspec = np.expand_dims(melspec, 0)
melspec = np.expand_dims(melspec, -1)
logspec = np.expand_dims(logspec, 0)
logspec = np.expand_dims(logspec, -1)

# print(mfcc.shape, logspec.shape)

# def getMELspectrogram(audio, sample_rate):
#     mel_spec = librosa.feature.melspectrogram(y=audio,
#                                               sr=sample_rate,
#                                               n_fft=1024,
#                                               win_length=512,
#                                               window='hamming',
#                                               hop_length=256,
#                                               n_mels=128,
#                                               fmax=sample_rate/2
#                                               )
#     mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
#     return mel_spec_db


def cnn_model():
    model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), input_shape=(13, 216, 1),
               activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(rate=0.2),

        Conv2D(filters=32, kernel_size=(3, 3),
               activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(rate=0.2),


        Conv2D(filters=32, kernel_size=(3, 3),
               activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(rate=0.2),


        Conv2D(filters=32, kernel_size=(3, 3),
               activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(rate=0.2),

        Flatten()

    ])

    return model


model_1 = cnn_model()


def model_2():

    cnn_input = Input(shape=(40, 256, 1))

    cnn_output = Conv2D(filters=32, kernel_size=(3, 3),
                        activation='relu', padding='same')(cnn_input)

    cnn_output = BatchNormalization()(cnn_output)
    cnn_output = Dropout(rate=0.2)(cnn_output)

    cnn_output = Conv2D(filters=32, kernel_size=(3, 3),
                        activation='relu')(cnn_output)
    cnn_output = BatchNormalization()(cnn_output)
    cnn_output = MaxPooling2D()(cnn_output)
    cnn_output = Dropout(rate=0.2)(cnn_output)

    cnn_output = Conv2D(filters=32, kernel_size=(3, 3),
                        activation='relu', padding='same')(cnn_output)
    cnn_output = BatchNormalization()(cnn_output)
    cnn_output = Dropout(rate=0.2)(cnn_output)

    cnn_output = Conv2D(filters=32, kernel_size=(3, 3),
                        activation='relu')(cnn_output)
    cnn_output = BatchNormalization()(cnn_output)
    cnn_output = MaxPooling2D()(cnn_output)
    cnn_output = Dropout(rate=0.2)(cnn_output)

    cnn_output = Flatten()(cnn_output)

    output = Dense(200, activation='relu')(cnn_output)
    output = Dense(10, activation='softmax')(output)

    model = Model(cnn_input, output)

    return model


m = model_2()

print(m(mfcc))
