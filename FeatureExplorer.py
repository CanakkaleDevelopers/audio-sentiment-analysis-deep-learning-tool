import os
import shutil


class FeatureExplorer:

    def __init__(self, path_dict):
        self.path_dict = path_dict

    def save_from_temp(self, features, title, note=""):
        """Features,baslik,not ile ağırlıkları Features dizini altında tutar"""
        os.mkdir(os.path.join(self.path_dict['FEATURES_FOLDER'], title))
        save_path = os.path.join(self.path_dict['FEATURES_FOLDER'], title)
        X = os.path.join(self.path_dict['TEMP_FOLDER'], 'FeaturesX.npy')
        Y = os.path.join(self.path_dict['TEMP_FOLDER'], 'FeaturesY.npy')

        dest_X = os.path.join(save_path, 'FeaturesX.npy')
        dest_Y = os.path.join(save_path, 'FeaturesY.npy')

        shutil.copy2(X, dest_X)
        shutil.copy2(Y, dest_Y)

        info_txt = os.path.join(save_path, 'info.txt')

        strFeatures = ' '.join([str(elem) for elem in features])
        Features = strFeatures
        Note = Features + '\n' + note

        f = open(info_txt, "w+")
        f.write(Note)
        f.close()

    def list_features(self):
        dirs = os.listdir(self.path_dict['FEATURES_FOLDER'])
        print('Bulunan Featurelar {}'.format(dirs))



        features = []
        for (root,dirs,files) in os.walk(self.path_dict['FEATURES_FOLDER']):
            context_dict = {}
            if 'info.txt' in files:
                f = open(os.path.join(self.path_dict['FEATURES_FOLDER'],'info.txt'))
                features = f.readline()
                notes = f.readline()
                #TODO -> burayı doldur







