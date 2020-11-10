import keras
from keras import layers


class ModelBuilder:

    @staticmethod
    def get_compiled_model(uncompiled_model, compile_config):
        """
        builds the model from json file

        Inputs:

        Outputs:
        Keras model (model)

        Saves:
        Keras model (h5py)
        Keras model (JSON)
        """

        model = uncompiled_model.compile(
            optimizer=compile_config['optimizer'],
            loss=compile_config['loss'],
            metrics=compile_config['metrics'],
        )
        return model

    @staticmethod()
    def get_uncompiled_model(Layers, input_shape):
        print("Building model flow")
        """
        Function initializes layers
        Inputs:
        layers (dict) {'layer_name':{'layer_parameters'}}
        :return:
        model (keras model)
        """

        model = keras.Sequential()

        for layer_name, param in Layers:

            if layer_name == 'Input':
                print("Input layer added.")
                input_layer = keras.layers.InputLayer(
                    input_shape=input_shape, batch_size=None, dtype=None, input_tensor=None, sparse=False,
                    name=None, ragged=False
                )


            elif layer_name == 'Conv1D':
                print("Conv1D layer added.")
                conv1_d_layer = layers.Conv1D(
                    param['filters'],
                    param['kernel_size'],
                    strides=param['strides'],
                    padding=param['padding'],
                    data_format=param['data_format'],
                    dilation_rate=param['dilation_rate'],
                    groups=param['groups'],
                    activation=param['activation'],
                    use_bias=param['use_bias'],
                    kernel_initializer=param['kernel_initializer'],
                    bias_initializer=param['bias_initializer'],
                    kernel_regularizer=param['kernel_regularizer'],
                    bias_regularizer=param['bias_regularizer'],
                    activity_regularizer=param['activity_regularizer'],
                    kernel_constraint=param['kernel_constraint'],
                    bias_constraint=param['bias_constraint']
                )
                model.add(conv1_d_layer)

            elif layer_name == 'Conv2D':
                print("Conv2D layer added.")

                conv2_d_layer = layers.Conv2D(
                    param['filters'],
                    param['kernel_size'],
                    strides=param['strides'],
                    padding=param['padding'],
                    data_format=param['data_format'],
                    dilation_rate=param['dilation_rate'],
                    groups=param['groups'],
                    activation=param['activation'],
                    use_bias=param['use_bias'],
                    kernel_initializer=param['kernel_initializer'],
                    bias_initializer=param['bias_initializer'],
                    kernel_regularizer=param['kernel_regularizer'],
                    bias_regularizer=param['bias_regularizer'],
                    activity_regularizer=param['activity_regularizer'],
                    kernel_constraint=param['kernel_constraint'],
                    bias_constraint=param['bias_constraint']
                )
                model.add(conv2_d_layer)
            elif layer_name == 'Dropout':
                print("Dropout layer added.")
                dropout_layer = keras.layers.Dropout(param['rate'], noise_shape=param['noise_shape']
                                                     , seed=param['seed'])
                model.add(dropout_layer)
            elif layer_name == 'BatchNormalization':
                print("BatchNormalization layer added.")
                batch_normalization_layer = layers.BatchNormalization(
                    axis=param['axis'],
                    momentum=param['momentum'],
                    epsilon=param['epsilon'],
                    center=param['center'],
                    scale=param['scale'],
                    beta_initializer=param['beta_initializer'],
                    gamma_initializer=param['gamma_initializer'],
                    moving_mean_initializer=param['moving_mean_initializer'],
                    moving_variance_initializer=param['moving_variance_initializer'],
                    beta_regularizer=param['beta_regularizer'],
                    gamma_regularizer=param['gamma_regularizer'],
                    beta_constraint=param['beta_constraint'],
                    gamma_constraint=param['gamma_constraint'],
                    renorm=param['renorm'],
                    renorm_clipping=param['renorm_clipping'],
                    renorm_momentum=param['renorm_momentum'],
                    fused=param['fused'],
                    trainable=param['trainable'],
                    virtual_batch_size=param['virtual_batch_size'],
                    adjustment=param['adjustment'],
                    name=param['name']
                )
            elif layer_name == 'Dense':
                print("Dense layer added.")
                dense_layer = keras.layers.Dense(
                    param['units'],
                    activation=param['activation'],
                    use_bias=param['use_bias'],
                    kernel_initializer=param['kernel_initializer'],
                    bias_initializer=param['bias_initializer'],
                    kernel_regularizer=param['kernel_regularizer'],
                    bias_regularizer=param['bias_regularizer'],
                    activity_regularizer=param['activity_regularizer'],
                    kernel_constraint=param['kernel_constraint'],
                    bias_constraint=param['bias_constraint']
                )
                model.add(dense_layer)
            elif layer_name == 'Flatten':
                flatten_layer = keras.layers.Flatten(data_format=param['data_format'])
                model.add(flatten_layer)
            else:
                print("unknow layer, process still going on.")

        return model
