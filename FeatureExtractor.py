import librosa
import numpy as np
import pandas as pd


class FeatureExtractor:

    # todo -> add normalize = True
    @staticmethod
    def read_audio(conf, pathname, trim_long_data=False):
        y, sr = librosa.load(pathname, sr=conf.sampling_rate)

        if 0 < len(y):
            y, _ = librosa.effects.trim(y)

        if len(y) > conf.samples:
            if trim_long_data:
                y = y[0:0 + conf.samples]
        else:
            padding = conf.samples - len(y)
            offset = padding // 2
            y = np.pad(y, (offset, conf.samples - len(y) - offset), 'constant')
        return y

    @staticmethod
    def extract(file_path, *args, normalize=False):

        """
        Ses dosyasının özniteliklerini döndürür.

        Parameters:

        file_path (string) : dosyanın yolu

        *args (list) : [mfcc,chroma,zcr,mel,contrast,tonnetz]

        mfccs (numpy.array) : mel Mel frekans ölçeği, insan kulağının ses frekanslarındaki değişimi algılayışını gösteren bir ölçektir.
        chroma (numpy.array) : Spektrum müzikal oktavının 12 farklı yarı tonunu(chroma) temsil eden 12 parçanın belirtildiği ses için güçlü bir sunumudur.
        mel (numpy.array) : mel spektogram verisi
        contrast (numpy.array) :
        tonnetz (numpy.array) :

        Returns :

        extracted_features (arr) : list of data
        lenght (int) : lenght of extracted_features array data

        Example use :

        FeatureExtractor.extract("example_audio.ogg",'mfcc','chroma') ||
        FeatureExtractor.extract("example_audio.ogg",'mfcc', normalize = True )

        """

        data, sample_rate = librosa.load(file_path)
        data = (data[:, 0] if data.ndim > 1 else data.T)

        # Get features

        stft = np.abs(librosa.stft(data))
        if "mfcc" in args: mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40).T,
                                          axis=0)  # 40 values
        if "chroma" in args: chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        if "mel" in args: mel = np.mean(librosa.feature.melspectrogram(data, sr=sample_rate).T, axis=0)
        if "contrast" in args: contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
        if "tonnetz" in args: tonnetz = np.mean(
            librosa.feature.tonnetz(y=librosa.effects.harmonic(data), sr=sample_rate).T,  # tonal centroid features
            axis=0)

        # öznitelik dizimizin uzunlugunu hesaplayalım

        extracted_features = []
        if 'mfcc' in locals(): extracted_features = np.hstack([extracted_features, mfcc])
        if 'chroma' in locals(): extracted_features = np.hstack([extracted_features, chroma])
        if 'mel' in locals(): extracted_features = np.hstack([extracted_features, mel])
        if 'contrast' in locals(): extracted_features = np.hstack([extracted_features, contrast])
        if 'tonnetz' in locals(): extracted_features = np.hstack([extracted_features, tonnetz])

        lenght = len(extracted_features)

        return extracted_features, lenght

    @staticmethod
    def extractSpectogram(file_path, conf, save=False):
        """
        Mel spektogram görüntüsünü döndürür.
        """
        audio = FeatureExtractor.read_audio()
        spectrogram = librosa.feature.melspectrogram(audio,
                                                     sr=conf.sampling_rate,
                                                     n_mels=conf.n_mels,
                                                     hop_length=conf.hop_length,
                                                     n_fft=conf.n_fft,
                                                     fmin=conf.fmin,
                                                     fmax=conf.fmax)
        spectrogram = librosa.power_to_db(spectrogram)
        spectrogram = spectrogram.astype(np.float32)

        return spectrogram
