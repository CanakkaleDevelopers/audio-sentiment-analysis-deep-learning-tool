import numpy as np
import random
import itertools
import librosa
import matplotlib.pyplot as plt
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
        sample_rate = conf.PreproccessConfig.sampling_rate
        pitch_change = conf.DataAugmentationConfig.pitch_pm * 2 * (np.random.uniform()-0.5)
        bins_per_octave = conf.DataAugmentationConfig.bins_per_octave
        data = librosa.effects.pitch_shift(data, sr=sample_rate, n_steps=pitch_change, bins_per_octave=bins_per_octave)
        return data

    @staticmethod

    def change_speed(data):
        speed_change = conf.DataAugmentationConfig.speed_change
        data = librosa.effects.time_stretch(data, speed_change)

    @staticmethod
    def remove_silent_regions(data):
        data, index = librosa.effects.trim(data)










    






