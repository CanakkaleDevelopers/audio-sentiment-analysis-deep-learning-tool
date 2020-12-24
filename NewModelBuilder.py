import keras
import numpy as np
import os


class NewModelBuilder:
    latest_model = None

    def __init__(self, path_dict, model_layers, compile_config):
        self.path_dict = path_dict
        self.model_layers = model_layers
        self.compile_config = compile_config

    def get_uncompiled_model(self):
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
        layer_type = ''
        for layer in self.model_layers:
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

        return model

    def get_compiled_model(self, uncompiled_model):
        print('Model, {} optimizeri, {} kayıp fonksiyonu ile derleniyor.'.format(self.compile_config['optimizer'],
                                                                                 self.compile_config['loss']))

        uncompiled_model.compile(optimizer=self.compile_config['optimizer'],
                                 loss=self.compile_config['loss'],
                                 metrics=self.compile_config['metrics'])

        model = uncompiled_model

        return model

    def build(self):
        model = self.get_uncompiled_model()
        model = self.get_compiled_model(model)

        model_path = os.path.join(self.path_dict['TEMP_FOLDER'], 'runtime_model')
        print("Model TEMP klasörünün altına oluşturuldu")
        model.save(model_path)
