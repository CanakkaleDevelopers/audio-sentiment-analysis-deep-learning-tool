class Config:

    class PreproccessConfig:
        sampling_rate = 44100
        duration = 2
        hop_length = 347 * duration  # to make time steps 128
        fmin = 20
        fmax = sampling_rate // 2
        n_mels = 128
        n_fft = n_mels * 20
        samples = sampling_rate * duration

        @classmethod
        def get_preproccess_conf(cls):
            conf_dict = {"sampling_rate": cls.sampling_rate,
                         "duration": cls.duration,
                         "hop_lenght": cls.hop_length,
                         "fmin": cls.fmin,
                         "fmax": cls.fmax,
                         "n_mels": cls.n_mels,
                         "n_fft": cls.n_fft,
                         "samples": cls.samples}

    class FilePathConfig:
        import sys
        import pathlib

        working_dir_path = pathlib.Path().absolute()

        if sys.platform.startswith('win32'):
            TRAINING_FILES_PATH = str(working_dir_path) + '\\pass\\'
            SAVE_DIR_PATH = str(working_dir_path) + '\\pass\\'
            MODEL_DIR_PATH = str(working_dir_path) + '\\pass\\'
            TEST_FILES_PATH = str(working_dir_path) + '\\pass\\'
        else:
            TRAINING_FILES_PATH = str(working_dir_path) + '/pass/'
            SAVE_DIR_PATH = str(working_dir_path) + '/pas/'
            MODEL_DIR_PATH = str(working_dir_path) + '/pass/'
            TEST_FILES_PATH = str(working_dir_path) + '/pass/'
