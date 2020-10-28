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
    def extract(file_path, features, augmentation=None, normalize=False):

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
        if 'white_noise' in augmentation:
            data = DataAugmentator.add_white_noise(data)
        elif 'stretch' in augmentation:
            data = DataAugmentator.stretch(data)
        elif 'shift' in augmentation:
            data = DataAugmentator.shift(data)
        elif 'change_speed' in augmentation:
            data = DataAugmentator.change_speed(data)
        else:
            pass

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
    def extractSpectogram(file_path, augmentation=None, save=False, show_in_console=False):
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

        # eğer data augmentation varsa veriye manipüle et yoksa devam
        if 'white_noise' in augmentation:
            audio = DataAugmentator.add_white_noise(audio)
        elif 'stretch' in augmentation:
            audio = DataAugmentator.stretch(audio)
        elif 'shift' in augmentation:
            audio = DataAugmentator.shift(audio)
        elif 'change_speed' in augmentation:
            audio = DataAugmentator.change_speed(audio)
        else:
            pass

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
    def extract_ravdess():
        import os

        root_path = conf.FilePathConfig.RAVDESS_FILES_PATH

        save_path = conf.FilePathConfig.SAVE_DIR_PATH
        features_list = []
        for subdir, dirs, files in os.walk(root_path):
            for file in files:
                try:
                    features = \
                        FeatureExtractor.extract(os.path.join(subdir, file), conf.PreproccessConfig.desired_features)[
                            0]  # 0-> feature 1 -> lenght of arr
                    emotion_code = int(file[7:8]) - 1  # 0-7 emotions
                    arr = features, emotion_code
                    features_list.append(arr)
                    print(arr)
                # dosya adı vs yanlıs olması veya islenememesi durumunda işlem durmamalı
                except ValueError as err:
                    print(err)
                    continue

        X, y = zip(*features_list)

        X, y = np.asarray(X), np.asarray(y)

        print(X.shape, y.shape)

        x_name, y_name = 'ravdessX.joblib', 'ravdessY.joblib'

        import joblib  # disk dump

        joblib.dump(X, os.path.join(save_path, x_name))
        joblib.dump(y, os.path.join(save_path, y_name))


""
