import numpy as np
import random
import itertools
import librosa
import matplotlib.pyplot as plt


class DataAugmentator:
    def __init__(self, data_augmentation_config):
        self.conf = data_augmentation_config

    """Bu sınıf elimizdeki veri setinin boyutunu büyütmek amacı ile
    Ses verisine bir takım işlemler uygularak yeni ses dosyaları oluşturur."""

    def add_white_noise(self, data):
        # karıncalanma gürültüsü ekler
        wn = np.random.randn(len(data))
        data_wn = data + 0.005 * wn
        return data_wn

    def shift(self, data):
        "ses dosyasının frekansını kaydırır"
        data_roll = np.roll(data, self.conf['shift_rate'])
        return data_roll

    def stretch(self, data):
        """ses verisini gererek genişletir"""
        sample_rate = 44100
        pitch_change = self.conf['pitch_pm'] * 2 * (np.random.uniform() - 0.5)
        bins_per_octave = self.conf['bins_per_octave']
        data = librosa.effects.pitch_shift(data, sr=sample_rate, n_steps=pitch_change, bins_per_octave=bins_per_octave)
        return data

    def change_speed(self, data):
        speed_change = self.conf['speed_change']
        data = librosa.effects.time_stretch(data, speed_change)
        return data



















