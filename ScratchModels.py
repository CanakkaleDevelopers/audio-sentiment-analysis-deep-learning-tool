from Config import Config
from keras import layers
import joblib
import os
import numpy as  np
from sklearn.model_selection import train_test_split


class ScratchModels:
    @staticmethod
    def train_model_1():
        "Ravdes ve tess veri seti ile mfcc özniteliği üzerinden eğitim yapar "
        X_file_path = os.path.join(Config.FilePathConfig.SAVE_DIR_PATH,'X.joblib')
        Y_file_path = os.path.join(Config.FilePathConfig.SAVE_DIR_PATH, 'Y.joblib')

        X = joblib.load(X_file_path)
        y = joblib.load(Y_file_path)

        X_train, X_test, y_train, y_test = train_test_split(X, y, Config.ModelTrainingConfig.train_test_split_rate,
                                                            Config.ModelTrainingConfig.random_state)

        x_traincnn = np.expand_dims(X_train, axis=2)
        x_testcnn = np.expand_dims(X_test, axis=2)

        print(x_traincnn.shape,x_testcnn.shape)




