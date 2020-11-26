import numpy as np
import os
class ModelTrainer:
    def __init__(self, model_train_config, path_dict):
        self.model_train_config = model_train_config
        self.path_dict = path_dict

    def train_with_temp_features(self):

        X = np.load(os.path.join(self.path_dict['TEMP_FOLDER'],'FeaturesX.npy'))
        Y = np.load(os.path.join(self.path_dict['TEMP_FOLDER'],'FeaturesY.npy'))


