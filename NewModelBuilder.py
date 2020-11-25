import keras
from keras import layers
input_layer = {'name': 'input_layer', 'input_shape': (40, 1),'batch_size':10}
conv_1d = {'name': 'conv_1d', 'filters': 32, 'kernel_size': (3),'padding':'same'}

model_layers = [input_layer,conv_1d,conv_1d]

model = keras.Sequential()
for layer in model_layers:
    if layer['name'] == 'input_layer':
        new_layer = keras.layers.InputLayer(input_shape=layer['input_shape'], batch_size= layer['batch_size'])
    elif layer['name'] == 'conv_1d':
        new_layer = keras.layers.Conv1D(filters=layer['filters'],kernel_size=layer['kernel_size'])

    model.add(new_layer)


