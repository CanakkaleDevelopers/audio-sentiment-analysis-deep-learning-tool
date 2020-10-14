def main():
    # deneme
    from FeatureExtractor import FeatureExtractor
    print(FeatureExtractor.extractSpectogram("example_audio.ogg").shape)
if __name__ == "__main__":
    main()
