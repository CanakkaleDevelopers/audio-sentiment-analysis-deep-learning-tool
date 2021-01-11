import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tensorflow.keras.models import load_model
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

    def train_with_temp_features(self):
        model_path = os.path.join(self.path_dict['SAVE_RUNTIME_FEATURES'], 'runtime_model')
        compiled_model = load_model(model_path, compile=False)
        import pickle
        compile_config = pickle.load(open("compile_config.p", "rb"))
        compiled_model.compile(optimizer=compile_config['optimizer'],
                               loss=compile_config['loss'],
                               metrics=['accuracy'])
        X_tum, Y_tum = np.load('TEMP/FeaturesX.npy'), np.load('TEMP/FeaturesY.npy')

        dic = {0: 'neutral', 1: 'happy', 2: 'sad', 3: 'angry', 4: 'fear', 5: 'disgust', 6: 'surprise', 7: 'bored'}

        labels = []
        for i in range(len(Y_tum)):
            label = Y_tum[i]
            if 'neutral' in label:
                labels.append(0)
            elif 'happy' in label:
                labels.append(1)
            elif 'sad' in label:
                labels.append(2)
            elif 'angry' in label:
                labels.append(3)
            elif 'fear' in label:
                labels.append(4)
            elif 'disgust' in label:
                labels.append(5)
            elif 'surprise' in label:
                labels.append(6)
            elif 'bored' in label:
                labels.append(7)

        Y_tum = np.asarray(labels)

        X_tum = np.expand_dims(X_tum, -1)
        Y_tum = np.expand_dims(Y_tum, -1)
        from sklearn.model_selection import train_test_split

        if self.use_random_state:

            X_tum_train, X_tum_test, Y_tum_train, Y_tum_test = train_test_split(X_tum, Y_tum,
                                                                                test_size=self.test_split_rate,
                                                                                random_state=42)
        else:
            X_tum_train, X_tum_test, Y_tum_train, Y_tum_test = train_test_split(X_tum, Y_tum,
                                                                                test_size=self.test_split_rate)

        print('{} oranında test ve train olarak bölündü.'.format(self.test_split_rate))

        print("Tensorboard hazırlanıyor..")
        print("DİKKAT! Tensorboard'ın programca açılması için programı yetkili kullanıcı olarak başlatmayı unutmayın.")
        log_dir = self.path_dict['SAVE_RUNTIME_FEATURES'] + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', os.path.abspath(log_dir)])
        url = tb.launch()

        webbrowser.open(url)
        print("Eğitim başlıyor")
        compiled_model.fit(X_tum_train, Y_tum_train, validation_split=self.validation_split_rate, epochs=self.epochs,
                           shuffle=True,
                           batch_size=self.batch_size,
                           callbacks=[tensorboard_callback])

        compiled_model.save('TEMP/model.h5')
        y_pred = compiled_model(X_tum_test)
        y_pred_classes = np.argmax(y_pred, axis=-1)
        real_classes = np.squeeze(Y_tum_test)

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(real_classes, y_pred_classes)
        print(cm)

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
