import keras
import numpy as np
import os
from keras import layers

# input ve output layerleri burda verilmeyecek
conv_1d = {'type': 'conv_1d', 'filters': 32, 'kernel_size': 3, 'padding': 'same'}
dropout = {'type': 'dropout', 'rate': 0.5}
dense = {'type': 'dense', 'units': 32, 'activation': 'relu'}
batch_normalization = {'type': 'batch_normalization'}

my_layers = [conv_1d, dropout, dense]


class NewModelBuilder:
    latest_model = None

    def __init__(self, path_dict):
        self.path_dict = path_dict

    def get_uncompiled_model(self, model_layers):
        print('Model yapısı oluşturuluyor')

        model = keras.Sequential()
        # add input layer

        X = np.load(os.path.join(self.path_dict['TEMP_FOLDER'], 'FeaturesX.npy'))
        Y = np.load(os.path.join(self.path_dict['TEMP_FOLDER'], 'FeaturesY.npy'))

        X = X[:, :, np.newaxis]

        input_shape = X.shape[1:]
        output_shape = Y.shape[0]

        print(input_shape)
        print(output_shape)

        model.add(keras.layers.InputLayer(input_shape=X.shape[1:]))  # input shape
        layer_type=''
        for layer in model_layers:
            layer_type = layer['type'] 
            del layer['type']
            
            if layer_type == 'conv_1d':                
                new_layer = keras.layers.Conv1D(**layer)
            elif layer_type == 'dropout':
                print('Dropout katmanı eklendi.')
                new_layer = keras.layers.Dropout(**layer)
            elif layer_type == 'dense':
                print('Dense katmanı eklendi.')
                new_layer = keras.layers.Dense(**layer)
            elif layer_type == 'batch_normalization':
                print('Batch Normalization katmanı eklendi.')
                new_layer = keras.layers.BatchNormalization()
            elif layer_type == 'flatten':
                print('Flatten katmanı eklendi..')
                new_layer = keras.layers.Flatten()
            else:
                print('Unreckognized layer')
                continue

            model.add(new_layer)

        model.add(keras.layers.Dense(output_shape, activation='softmax'))  # output shape

        self.latest_model = model
        return model

    def get_compiled_model(self, compile_config, uncompiled_model):
        print('Model, {} optimizeri, {} kayıp fonksiyonu ile derleniyor.'.format(compile_config['optimizer'],
                                                                                 compile_config['loss']))

        uncompiled_model.compile(optimizer=compile_config['optimizer'],
                                 loss=compile_config['loss'],
                                 metrics=compile_config['metrics'])

        model = uncompiled_model

        return model
