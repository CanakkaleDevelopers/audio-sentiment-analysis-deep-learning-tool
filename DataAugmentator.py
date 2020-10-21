import numpy as np
import random
import itertools
import librosa
import matplotlib.pyplot as plt
from FeatureExtractor import FeatureExtractor  as FE
from Config import Config as conf
class DataAugmentator:
    """Bu sınıf elimizdeki veri setinin boyutunu büyütmek amacı ile
    Ses verisine bir takım işlemler uygularak yeni ses dosyaları oluşturur."""

    @staticmethod
    def plot_time_series(data):
        fig = plt.figure(figsize=(14, 8))
        fig = plt.figure(figsize=(14, 8))
        plt.title('Raw wave ')
        plt.ylabel('Amplitude')
        plt.plot(np.linspace(0, 1, len(data)), data)
        plt.show()

    @staticmethod
    def add_white_noise(data):
        # karıncalanma gürültüsü ekler
        wn = np.random.randn(len(data))
        data_wn = data + 0.005 * wn
        return data_wn

    @staticmethod
    def shift(data):
        "ses dosyasının frekansını kaydırır"
        data_roll = np.roll(data, conf.DataAugmentationConfig.shift_rate)
        return data_roll

    @staticmethod
    def stretch(data):
        """ses verisini gererek genişletir"""
        data = librosa.effects.time_stretch(data, conf.DataAugmentationConfig.strectch_rate)
        return data





    






