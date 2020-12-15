import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


class ModelTrainer:
    def __init__(self, model_train_config, path_dict):
        self.split_rate = model_train_config['test_split_rate']
        self.batch_size = model_train_config['batch_size']
        self.use_random_state = model_train_config['use_random_state']
        self.path_dict = path_dict

    def train_with_temp_features(self, compiled_model=None):

        X = np.load(os.path.join(self.path_dict['TEMP_FOLDER'], 'FeaturesX.npy'))
        Y = np.load(os.path.join(self.path_dict['TEMP_FOLDER'], 'FeaturesY.npy'))

        label_dict, Y = self.string_labels_to_categorical(Y)

        if self.use_random_state:
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=self.split_rate, random_state=42)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=self.split_rate)

        print('{} oranında test ve train olarak bölündü.'.format(self.split_rate))

        print('Etiketler ve eğitimdeki Değerleri')
        print(json.dumps(label_dict, indent=1))

        print("Eğitim başlıyor")

        compiled_model

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
