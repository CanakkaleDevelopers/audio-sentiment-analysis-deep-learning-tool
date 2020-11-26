import keras
from keras import layers

input_layer = {'name': 'input_layer', 'input_shape': (40, 1), 'batch_size': (40)}
conv_1d = {'name': 'conv_1d', 'filters': 32, 'kernel_size': 3, 'padding': 'same'}
dropout = {'name': 'dropout', 'rate': 0.5}
dense = {'name': 'dense', 'units': 32, 'activation': 'relu'}
batch_normalization = {'name': 'batch_normalization'}

my_layers = [input_layer, conv_1d, dropout, dense]


class NewModelBuilder():
    latest_model = None


    def get_uncompiled_model(self, model_layers):
        print('Model yapısı oluşturuluyor')
        model = keras.Sequential()
        for layer in model_layers:
            if layer['name'] == 'input_layer':
                print('giriş katmanı eklendi.')
                new_layer = keras.layers.InputLayer(input_shape=layer['input_shape'])
            elif layer['name'] == 'conv_1d':
                print('Conv_1D katmanı eklendi.')
                new_layer = keras.layers.Conv1D(filters=layer['filters'], kernel_size=layer['kernel_size'],activation=layer['activation'])
            elif layer['name'] == 'dropout':
                print('Dropout katmanı eklendi.')
                new_layer = keras.layers.Dropout(rate=layer['rate'])
            elif layer['name'] == 'dense':
                print('Dense katmanı eklendi.')
                new_layer = keras.layers.Dense(layer['units'], activation=layer['activation'])
            elif layer['name'] == 'batch_normalization':
                print('Batch Normalization katmanı eklendi.')
                new_layer = keras.layers.BatchNormalization()
            elif layer['name'] == 'flatten':
                print('Flatten katmanı eklendi..')
                new_layer = keras.layers.Flatten()
            model.add(new_layer)
        self.latest_model = model
        return model

    def get_compiled_model(self, compile_config):
        print('Model, {} optimizeri, {} kayıp fonksiyonu ile derleniyor.'.format(compile_config['optimizer'],compile_config['loss']))
        model = self.latest_model
        model = model.compile(optimizer=compile_config['optimizer'],
                              loss=compile_config['loss'],
                              metrics=compile_config['metrics'])
        return model


