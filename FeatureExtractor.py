import librosa
import numpy as np
import pandas as pd
from Config import Config as conf
import skimage.io
import os
import warnings

from DataAugmentator import DataAugmentator


class FeatureExtractor:

    # todo -> add normalize = True
    @staticmethod
    def read_audio(pathname, trim_long_data=False):
        y, sr = librosa.load(pathname, sr=conf.PreproccessConfig.sampling_rate)

        if 0 < len(y):
            y, _ = librosa.effects.trim(y)

        if len(y) > conf.PreproccessConfig.samples:
            if trim_long_data:
                y = y[0:0 + conf.PreproccessConfig.samples]
        else:
            padding = conf.PreproccessConfig.samples - len(y)
            offset = padding // 2
            y = np.pad(y, (offset, conf.PreproccessConfig.samples - len(y) - offset), 'constant')
        return y

    @staticmethod
    def extract(file_path, features, normalize=False):

        """
        Ses dosyasının özniteliklerini döndürür.

        Parameters:

        file_path (string) : dosyanın yolu

        features (list) : [mfcc,chroma,zcr,mel,contrast,tonnetz] len(features) => 1 olmalı

        augmentation (list) : ['white_noise' , 'stretch','shift'] , default : None


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

        if len(features) == 0:
            print("You need to extract at least one feature")
            return

        data = FeatureExtractor.read_audio(file_path)
        # data , _ = librosa.load(file_path,sr=conf.PreproccessConfig.sampling_rate)
        data = (data[:, 0] if data.ndim > 1 else data.T)

        # eğer data augmentation varsa veriye manipüle et yoksa devam
        data = FeatureExtractor.add_augmentation_da_data_helper(data)

        # Get features
        sample_rate = conf.PreproccessConfig.sampling_rate
        stft = np.abs(librosa.stft(data))
        if "mfcc" in features: mfcc = np.mean(
            librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=conf.PreproccessConfig.n_mfcc).T,
            axis=0)  # 40 values
        if "chroma" in features: chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        if "mel" in features: mel = np.mean(librosa.feature.melspectrogram(data, sr=sample_rate).T, axis=0)
        if "contrast" in features: contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,
                                                      axis=0)
        if "tonnetz" in features: tonnetz = np.mean(
            librosa.feature.tonnetz(y=librosa.effects.harmonic(data), sr=sample_rate).T,  # tonal centroid features
            axis=0)

        if "mfcc_delta" in features: mfcc_delta = np.mean(
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
    def extractSpectogram(file_path, save=False):
        """
        Mel spektogram görüntüsünü config dosyasındaki spectogram konumuna kaydeder,
        görüntüler insan tarafından değil  makine tarafından okunmak adına, x,y label isimleri vs barındırmaz.
        dosyanın adı audio dosyasının kendi adı ve uzantısı 'png' dir.

        Parameters

        file_path (str) : dosyanın yolu
        augmentation (list) : ['white_noise' , 'stretch','shift'] , default : None

        Returns & Saves

        no return -void-

        spectogram file (file) : filename.desired_image_extension

        """
        warnings.filterwarnings("ignore")
        audio = FeatureExtractor.read_audio(file_path)

        audio = FeatureExtractor.add_augmentation_da_data_helper(audio)

        spectrogram = librosa.feature.melspectrogram(audio,
                                                     sr=conf.PreproccessConfig.sampling_rate,
                                                     n_mels=conf.PreproccessConfig.n_mels,
                                                     hop_length=conf.PreproccessConfig.hop_length,
                                                     n_fft=conf.PreproccessConfig.n_fft,
                                                     fmin=conf.PreproccessConfig.fmin,
                                                     fmax=conf.PreproccessConfig.fmax)
        spectrogram = np.log(spectrogram + 1e-9)  # add small number to avoid log(0)

        img = FeatureExtractor.scale_minmax(spectrogram, 0, 255).astype(np.uint8)
        img = np.flip(img, axis=0)  # put low frequencies at the bottom in image
        img = 255 - img  # invert. make black==more energy

        save_path = conf.FilePathConfig.TRAINING_FILES_SPECTOGRAMS
        file_name = os.path.basename(file_path).split('.')[0]  # file name without extension
        file_name = '{}.{}'.format(file_name,
                                   conf.PreproccessConfig.spectogram_file_extension)  # file name and extension default : png for write
        out = os.path.join(save_path, file_name)  # file name with png extension
        print(out)
        skimage.io.imsave(out, img)

    @staticmethod
    def scale_minmax(X, min=0.0, max=1.0):
        X_std = (X - X.min()) / (X.max() - X.min())
        X_scaled = X_std * (max - min) + min
        return X_scaled


    @staticmethod
    def extract_from_metadata_table():
        """
        Bu fonksiyon çalışma zamanında oluşturulan TEMP/metadata_table.csv dosyasını baz alarak, orada yolu bulunan dosyaların,
         özniteliklerini çıkartır, ve çalışma zamanında TEMP/Features.joblib dosyasını oluşturur.

         Inputs: none
         Dependencies: TEMP/metadata_table.csv && FeatureExtractor class methods && Config class
        :return:
        """
        path_conf = conf.FilePathConfig
        preprocess_conf = conf.PreproccessConfig
        metadata_df = pd.read_csv(path_conf.DATA_METADATA_DF_PATH)
        metadata_df = metadata_df.loc[:, ~metadata_df.columns.str.contains('^Unnamed')]

        features_x_path = os.path.join(path_conf.SAVE_RUNTIME_FEATURES, 'featuresX')
        features_y_path = os.path.join(path_conf.SAVE_RUNTIME_FEATURES, 'featuresY')

        _, feature_x_lenght = FeatureExtractor.extract('example_audio.ogg', preprocess_conf.desired_features)  # olusacak array shepini almak için

        features_x = np.empty(feature_x_lenght)
        features_y = []
        for index, row in metadata_df.iterrows():
            print("Emotion:{} Dataset:{} {}/{}".format(row['labels'], row['source'], index, len(metadata_df)))
            row_features, _ = FeatureExtractor.extract(row['path'], preprocess_conf.desired_features)
            len(row_features)
            features_x = np.vstack([features_x, row_features])
            features_y = np.hstack([features_y, row['labels']])

        np.save(features_x_path, features_x)
        np.save(features_y_path, features_y)



        print("Öznitelik çıkarım işlemi sonu")


    @staticmethod
    def add_augmentation_da_data_helper(data):
        """
        returns augmented data if
        :param data: sound data
        :param augmentation: augmentation list can be none
        :return: augmented or non augmented data
        """

        from Config import Config
        augmentation_config = Config.DataAugmentationConfig
        augmentation = augmentation_config.augmentations
        if augmentation_config.augment_data is not False:
            # eğer data augmentation varsa veriye manipüle et yoksa devam
            if 'white_noise' in augmentation:
                audio = DataAugmentator.add_white_noise(data)
            if 'stretch' in augmentation:
                audio = DataAugmentator.stretch(data)
            if 'shift' in augmentation:
                audio = DataAugmentator.shift(data)
            if 'change_speed' in augmentation:
                audio = DataAugmentator.change_speed(data)
            else:
                pass
        else:
            pass

        return data


