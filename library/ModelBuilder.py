import keras
import tensorflow as tf
import numpy as np
import os


class NewModelBuilder:
    latest_model = None

    def __init__(self, path_dict, model_layers, compile_config):
        self.path_dict = path_dict
        self.model_layers = model_layers
        self.compile_config = compile_config
        # print(self.path_dict)

    def get_uncompiled_model(self):
        print('Model yapısı oluşturuluyor')

        model = keras.Sequential()
        # add input layer
        X = np.load(os.path.join(self.path_dict['SAVE_RUNTIME_FEATURES'], 'FeaturesX.npy'))
        Y = np.load(os.path.join(self.path_dict['SAVE_RUNTIME_FEATURES'], 'FeaturesY.npy'))

        unique_elements = {}
        label_code = 0
        for i in Y:
            if i not in unique_elements:
                unique_elements[i] = label_code
                label_code += 1

        new_labels = []
        for i in Y:
            new_labels.append(unique_elements[i])

        Y = np.asarray(new_labels)
        # X = X[:, :, np.newaxis]
        Y = Y[:, np.newaxis]

        input_shape = X.shape[1:]
        output_shape = len(np.unique(Y))

        print(input_shape)
        print(output_shape)

        model.add(keras.layers.Input(shape=X.shape[1:]))  # input shape
        layer_type = ''
        for layerx in self.model_layers:
            layer = layerx.__dict__
            layer_type = layer['type']  # düzenlendi . Bu şekilde ulaşılabiliyor.
            del layer['type']
            del layer['id']
            del layer['_sa_instance_state']
            layer = {k: v for k, v in layer.items() if v is not None}

            if layer_type == 'conv_1d':
                new_layer = keras.layers.Conv1D(**layer)
                print('conv_1d katmanı eklendi.')
            elif layer_type == 'dropout':
                print('Dropout katmanı eklendi.')
                new_layer = keras.layers.Dropout(**layer)
            elif layer_type == 'dense':
                print('Dense katmanı eklendi.')
                new_layer = keras.layers.Dense(**layer)
            elif layer_type == 'batch_normalization':
                print('Batch Normalization katmanı eklendi.')
                new_layer = keras.layers.BatchNormalization()
            elif layer_type == 'max_pooling_1d':
                print('MaxPooling1D katmanı eklendi.')
                new_layer = keras.layers.MaxPooling1D()
            elif layer_type == 'flatten':
                print('Flatten katmanı eklendi..')
                new_layer = keras.layers.Flatten()
            else:
                print('Unreckognized layer')
                continue

            model.add(new_layer)

        model.add(keras.layers.Dense(output_shape, activation='softmax'))  # output shape
        print(model.summary())

        print('Model, {} optimizeri, {} kayıp fonksiyonu ile derleniyor.'.format(self.compile_config['optimizer'],
                                                                                 self.compile_config['loss']))

        model.compile(optimizer=self.compile_config['optimizer'],
                      loss=self.compile_config['loss'],
                      metrics=['accuracy'])

        import pickle
        pickle.dump(self.compile_config, open("compile_config.p", "wb"))

        return model

    def get_compiled_model(self, uncompiled_model):
        print('Model, {} optimizeri, {} kayıp fonksiyonu ile derleniyor.'.format(self.compile_config['optimizer'],
                                                                                 self.compile_config['loss']))

        uncompiled_model.compile(optimizer='adam',
                                 loss='sparse_categorical_crossentropy',
                                 metrics=['accuracy'])
        model = uncompiled_model

        return model

    def build(self):
        model = self.get_uncompiled_model()
        model_path = os.path.join(self.path_dict['SAVE_RUNTIME_FEATURES'], 'runtime_model')
        print("Model TEMP klasörünün altına oluşturuldu")
        tf.keras.models.save_model(model, model_path, save_format='h5')
        model.summary()
