import librosa
import numpy as np
import pandas as pd
from Config import Config as conf
import skimage.io
import os
import warnings

from DataAugmentator import DataAugmentator


class FeatureExtractor:
    def __init__(self, feature_extraction_dict, data_augmentation_dict):
        self.data_augmentator = DataAugmentator(data_augmentation_dict)
        self.sampling_rate = feature_extraction_dict['sampling_rate']
        self.duration = feature_extraction_dict['duration']
        self.samples = feature_extraction_dict['duration'] * feature_extraction_dict['sampling_rate']
        self.n_mfcc = feature_extraction_dict['n_mfcc']
        self.features = feature_extraction_dict['features']  # çıkartılacak featureler, liste olmalı
        self.augmentations = feature_extraction_dict['augmentations']
        self.trim_long_data = feature_extraction_dict['trim_long_data']

    # todo -> add normalize = True
    def read_audio(self, pathname):
        """
        sesi yükler ve sesin dosyasındaki sessiz bölgeleri atar,
        ardından ses kısaysa doldurur,ses uzunsa belirtilen süreden
        uzun kısmını keser.
        :param pathname:
        :return:
        """
        y, sr = librosa.load(pathname, sr=self.sampling_rate)

        if 0 < len(y):
            y, _ = librosa.effects.trim(y)

        if len(y) > self.samples:
            if self.trim_long_data:
                y = y[0:0 + self.samples]
        else:
            padding = self.samples - len(y)
            offset = padding // 2
            y = np.pad(y, (offset, self.samples - len(y) - offset), 'constant')
        return y

    def extract(self, file_path, make_augmentations=False):

        """
        Ses dosyasının özniteliklerini döndürür.

        Parameters:

        file_path (string) : dosyanın yolu

        make_augmentations : eğer true ise augmented data özellikleri döndürür


        mfccs (numpy.array) : mel Mel frekans ölçeği, insan kulağının ses frekanslarındaki değişimi algılayışını gösteren bir ölçektir.
        chroma (numpy.array) : Spektrum müzikal oktavının 12 farklı yarı tonunu(chroma) temsil eden 12 parçanın belirtildiği ses için güçlü bir sunumudur.
        mel (numpy.array) : mel spektogram verisi
        contrast (numpy.array) :
        tonnetz (numpy.array) :
        mfcc_delta (numpy.array) : girdi verilerinin türevinin yerel tahmini

        Returns :

        extracted_features (arr) : list of data
        lenght (int) : lenght of extracted_features array data

        Example use :

        FeatureExtractor.extract("example_audio.ogg",['mfcc''chroma']) ||
        FeatureExtractor.extract("example_audio.ogg",['mfcc'], normalize = True )

        """

        warnings.filterwarnings("ignore")

        if len(self.features) == 0:
            print("You need to extract at least one feature")
            return

        data = self.read_audio(file_path)
        data = (data[:, 0] if data.ndim > 1 else data.T)

        # eğer data augmentation varsa veriye manipüle et yoksa devam
        if make_augmentations:
            data = self.augment_data(data)

        # Get features
        sample_rate = self.sampling_rate
        stft = np.abs(librosa.stft(data))
        if "mfcc" in self.features: mfcc = np.mean(
            librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=self.n_mfcc).T,
            axis=0)  # 40 values
        if "chroma" in self.features: chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        if "mel" in self.features: mel = np.mean(librosa.feature.melspectrogram(data, sr=sample_rate).T, axis=0)
        if "contrast" in self.features: contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,
                                                           axis=0)
        if "tonnetz" in self.features: tonnetz = np.mean(
            librosa.feature.tonnetz(y=librosa.effects.harmonic(data), sr=sample_rate).T,  # tonal centroid features
            axis=0)

        if "mfcc_delta" in self.features: mfcc_delta = np.mean(
            librosa.feature.delta(librosa.feature.mfcc(y=data, sr=sample_rate)))

        # öznitelik dizimizin uzunlugunu hesaplayalım

        extracted_features = []
        if 'mfcc' in locals(): extracted_features = np.hstack([extracted_features, mfcc])
        if 'chroma' in locals(): extracted_features = np.hstack([extracted_features, chroma])
        if 'mel' in locals(): extracted_features = np.hstack([extracted_features, mel])
        if 'contrast' in locals(): extracted_features = np.hstack([extracted_features, contrast])
        if 'tonnetz' in locals(): extracted_features = np.hstack([extracted_features, tonnetz])
        if 'mfcc_delta' in locals(): extracted_features = np.hstack([extracted_features, mfcc_delta])

        lenght = len(extracted_features)

        return extracted_features, lenght

    @staticmethod
    def scale_minmax(X, min=0.0, max=1.0):
        X_std = (X - X.min()) / (X.max() - X.min())
        X_scaled = X_std * (max - min) + min
        return X_scaled

    def extract_with_database(self):
        # Before extraction loop variables decleration block
        save_features_X = 'TEMP/FeaturesX'  # temp_featuresX doya yolu
        save_features_Y = 'TEMP/FeaturesY'  # temp_featuresY dosya yolu
        _, features_x_lenght = self.extract('example_audio.ogg')  # dummy # burası değiştirilmememli
        features_x = np.empty(features_x_lenght)
        features_y = []

        # Looping all records block
        lenght_of_records_in_database = 20  # kayıtların toplam sayısı
        for count in range(lenght_of_records_in_database):  # -> veritabanındaki kayıt sayısı kadar dön

            # query record block
            record = {'Gender': 'Male', 'Emotion': 'Happy', 'Source': 'Ravdess', 'Path': 'example_audio.ogg',
                      'augment': True}  # burayı queryi yap sırayla oku qureyi yi bu şekilde getir veya alt tarafları düzenle

            print('Extracting selected features from  {} {} {} audio record. {} file left'.format(record['Source'],
                                                                                                  record['Emotion'],
                                                                                                  record['Gender'], (
                                                                                                          lenght_of_records_in_database - count)))

            # Extracting file himself
            record_features, _ = self.extract(record['Path'])
            record_label = record['Emotion']

            features_x = np.vstack([features_x, record_features])
            features_y = np.hstack([features_y, record_label])

            # Extracting augmented file if True
            if record['augment']:
                print('Extracting AUGMENTED record features from  {} {} {} audio record.'.format(
                    record['Source'],
                    record['Emotion'],
                    record['Gender'], ))
                record_features, _ = self.extract(record['Path'], make_augmentations=True)
                record_label = record['Emotion']

                features_x = np.vstack([features_x, record_features])
                features_y = np.hstack([features_y, record_label])

            # save block

            features_x_final = features_x[1:]  # trim first np.empty(40)
            np.save(save_features_X, features_x_final)
            np.save(save_features_Y, features_y)

            """
            if(save_this_features_forever):
                TODO-> eğer kullanıcı bu featureleri sonra da kullanmak isterse kaydedebilmeli
                çünkü işlem çok uzun
            """

    def augment_data(self, data):
        """
        returns augmented data if
        :param data: sound data

        :return: return augmented
        """

        augmentation = self.augmentations

        if 'white_noise' in augmentation:
            audio = self.data_augmentator.add_white_noise(data)
        if 'stretch' in augmentation:
            audio = self.data_augmentator.stretch(data)
        if 'shift' in augmentation:
            audio = self.data_augmentator.shift(data)
        if 'change_speed' in augmentation:
            audio = self.data_augmentator.change_speed(data)
        else:
            pass

        return audio
