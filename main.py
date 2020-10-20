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





if __name__ == "__main__":
    main()
