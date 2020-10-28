def main():
    # deneme
    from FeatureExtractor import FeatureExtractor
    # print(FeatureExtractor.extract("example_audio.ogg",['mfcc']))
    # print(FeatureExtractor.extract_ravdess())

    #from ScratchModels import ScratchModels

    #ScratchModels.train_model_1()
    import numpy as np
    import librosa

    data, sr = librosa.load("example_audio.ogg", sr=None)
    from sklearn.preprocessing import normalize as sklearnnormalize
    from librosa.util import normalize as libnormalize

    ##print(FeatureExtractor.extractSpectogram.__doc__)
    from DataAugmentator import DataAugmentator
    #DataAugmentator.shift(FeatureExtractor.read_audio('example_audio.ogg'))
    DataAugmentator.plot_time_series(DataAugmentator.stretch(FeatureExtractor.read_audio('example_audio.ogg')))



if __name__ == "__main__":
    main()
