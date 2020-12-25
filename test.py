from tensorflow.keras import Input, Model, Sequential, layers
from tensorflow.keras.layers import (LSTM, BatchNormalization, Concatenate,
                                     Conv2D, Conv1D, Dense, Dropout, Reshape, Flatten,
                                     MaxPooling2D, MaxPooling1D, Activation)
import numpy as np

# model = Sequential([
#     # ONEMLI: bunun activation parametresini vermedim, default olarak None, jsonda nasi gosterdigine bakmak icin. aynisi conv1d icinde gecerli onda da yazmazsan bir aktivasyon fonk uygulamiyor.
#     Conv2D(filters=32, kernel_size=(3, 3), input_shape=(10, 10, 3)),

#     BatchNormalization(),

#     Activation('relu'),  # bu da onemli

#     Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),

#     MaxPooling2D(),

#     Flatten(),

#     Dense(64, activation='relu'),

#     Dropout(0.2),

#     Dense(32,),

#     Dense(14, activation='softmax'),
# ])

model = Sequential([
    Conv2D(filters=64, kernel_size=(3, 1), input_shape=(10, 1)),


])

i = np.zeros((1, 10, 1))

print(model(i))
# Conv1D(filters=64, kernel_size=(3, ),
#        activation='relu'),
# Conv1D(filters=32, kernel_size=(3, )),

# MaxPooling1D(),

# with open('test.txt', 'w') as txt:
#     txt.write(model.to_json())
