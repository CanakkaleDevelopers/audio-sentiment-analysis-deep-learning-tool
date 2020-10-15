def main():
    # deneme
    from FeatureExtractor import FeatureExtractor
    ##print(FeatureExtractor.extractSpectogram("example_audio.ogg").shape)
    print(FeatureExtractor.extract("example_audio.ogg",['mfcc']))
    print(FeatureExtractor.extract_ravdess())
if __name__ == "__main__":
    main()
