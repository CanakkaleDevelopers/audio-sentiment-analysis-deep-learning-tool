class Config:
    class PreproccessConfig:
        sampling_rate = 44100
        duration = 2
        hop_length = 347 * duration  # to make time steps 128
        fmin = 20
        fmax = sampling_rate // 2
        n_mels = 128
        n_mfcc = 40
        n_fft = n_mels * 20
        samples = sampling_rate * duration
        desired_features = ['mfcc']
        is_Spectogram_selected = False
        spectogram_file_extension = 'png'

        @classmethod
        def feature_count(cls):
            return len(cls.selected_features)

    class DataAugmentationConfig:
        augment_data = False
        augmentations = ['addWhiteNoise','Shift']
        shift_rate = 1600
        strectch_rate = 1

    class FilePathConfig:
        import sys
        import pathlib

        working_dir_path = pathlib.Path().absolute()

        if sys.platform.startswith('win32'):
            TRAINING_FILES_PATH = str(working_dir_path) + '\\pass\\'
            TRAINING_FILES_SPECTOGRAMS = str(working_dir_path) + '\\ExtractedFeatures\\Spectogram\\'
            SAVE_DIR_PATH = str(working_dir_path) + '\\ExtractedFeatues\\'
            MODEL_DIR_PATH = str(working_dir_path) + '\\pass\\'
            TEST_FILES_PATH = str(working_dir_path) + '\\pass\\'
            RAVDESS_FILES_PATH = str(working_dir_path) + '\\Datasets\\Ravdess'
        else:
            TRAINING_FILES_PATH = str(working_dir_path) + '/pass/'
            TRAINING_FILES_SPECTOGRAMS = str(working_dir_path) + '/ExtractedFeatures/Spectogram'
            SAVE_DIR_PATH = str(working_dir_path) + '/ExtractedFeatues/'
            MODEL_DIR_PATH = str(working_dir_path) + '/pass/'
            TEST_FILES_PATH = str(working_dir_path) + '/pass/'
            RAVDESS_FILES_PATH = str(working_dir_path) + '/Datasets/Ravdess'

    class ModelTrainingConfig:
        use_pretrained_model = False
        use_split_random_state = False
        train_test_split_rate = 0.33
        random_state = 42 if use_split_random_state else None
