import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
import datetime
import os
import webbrowser
import threading
import time
from tensorboard import program


class ModelTrainer:
    def __init__(self, model_train_config, path_dict, tensorboard_config):
        self.validation_split_rate = model_train_config['validation_split_rate']
        self.epochs = model_train_config['epochs']
        self.test_split_rate = model_train_config['test_split_rate']
        self.batch_size = model_train_config['batch_size']
        self.use_random_state = model_train_config['use_random_state']
        self.path_dict = path_dict

    def train_with_temp_features(self, compiled_model):

        X = np.load(os.path.join(self.path_dict['TEMP_FOLDER'], 'FeaturesX.npy'))
        Y = np.load(os.path.join(self.path_dict['TEMP_FOLDER'], 'FeaturesY.npy'))

        X = X[:, :, np.newaxis]

        label_dict, Y = self.string_labels_to_categorical(Y)

        if self.use_random_state:
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=self.test_split_rate, random_state=42)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=self.test_split_rate)

        print('{} oranında test ve train olarak bölündü.'.format(self.test_split_rate))

        print('Etiketler ve eğitimdeki Değerleri')
        print(json.dumps(label_dict, indent=1))

        print("Tensorboard hazırlanıyor..")
        print("DİKKAT! Tensorboard'ın programca açılması için programı yetkili kullanıcı olarak başlatmayı unutmayın.")
        log_dir = self.path_dict['TENSORBOARD_LOGDIR'] + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', os.path.abspath(log_dir)])
        url = tb.launch()

        webbrowser.open(url)

        print("Eğitim başlıyor")
        compiled_model.fit(X, Y, validation_split=self.validation_split_rate, epochs=self.epochs, shuffle=True,
                           callbacks=[tensorboard_callback])

    def string_labels_to_categorical(self, labels):

        unique_elements = {}
        label_code = 0
        for i in labels:
            if i not in unique_elements:
                unique_elements[i] = label_code
                label_code += 1

        new_labels = []
        for i in labels:
            new_labels.append(unique_elements[i])

        new_labels = to_categorical(new_labels)

        return unique_elements, new_labels
