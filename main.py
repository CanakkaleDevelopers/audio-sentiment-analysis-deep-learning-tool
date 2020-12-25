def main():
    # deneme
    from FeatureExtractor import FeatureExtractor
    from DatasetExplorer import DatasetExplorer
    import numpy as np
    from DataMetaDataCreator import MetaDataCreator
    from NewModelBuilder import NewModelBuilder
    from ModelTrainer import ModelTrainer
    from FeatureExplorer import FeatureExplorer

    explore_datasets = False
    make_feature_extraction = False
    create_meta_csv = False
    select_pretrained_model = False
    build_your_model = True
    train_your_model = True
    use_feature_explorer = False

    """initializing neccesary vars"""
    path_dict = {'DOWNLOADS_FOLDER': 'Downloads', 'DATASETS_FOLDER': 'Datasets',
                 'emoDB': 'Datasets/emoDB', 'Ravdess': 'Datasets/Ravdess',
                 'SAVEE': 'Datasets/SAVEE', 'Crema-D': 'Datasets/Crema-D',
                 'TEMP_FOLD'
                 ''
                 'ER': 'TEMP', 'FEATURES_FOLDER': 'Features', 'MODELS_FOLDER': 'Models',
                 'TENSORBOARD_LOGDIR': 'Models/Tensorboard'}

    """0.5 inci faz veriseti tarama ve indirme"""
    if explore_datasets:
        demanded_datasets = ['emoDB']
        dataset_explorer = DatasetExplorer(demanded_datasets, path_dict)  # demanded_datasets[], path_dict{} yolla

        dataset_explorer.scan()  # locali tara
        dataset_explorer.download_datasets()  # bulunmayanları indir, bunun çalışması için üstteki f çalışmalı

    """1. Faz meta csv oluşturma"""
    if create_meta_csv:
        data_meta_data_creator = MetaDataCreator()  # path_dict{}, yolla
        data_meta_data_creator.create_csv()  # demanded_datasets[], yolla

    """
    1.5 Faz  -> 1. fazın yarattığı metadata_table.csv dosyasından kullanıcının isteklerine göre istenmeyen verileri sil,
    ve dataframei karıştır, tekrar aynı isimle ve aynı yere metadata_table.csv dosyasını kaydet.
    """

    """
    2. Faz -> Öznitelikleri çıkart ve TEMP/FeaturesX.npy ve TEMP/FeaturesY.npy dosyaları oluşsun.
    """
    if make_feature_extraction:
        # initialization of FeatureExtractor class block
        feature_extraction_dict = {'sampling_rate': 44100, 'duration': 4, 'trim_long_data': False, 'n_mfcc': 40,
                                   'features': ['mfcc', 'chroma'],
                                   'augmentations': ['white_noise', 'stretch', 'shift', 'change_speed']}
        data_augmentation_dict = {'shift_rate': 1600, 'stretch_rate': 1, 'speed_change': 1,
                                  'pitch_pm': 24, 'bins_per_octave': 24}
        f = FeatureExtractor(feature_extraction_dict, data_augmentation_dict)  # feature_extraction_dict {} yolla,

        f.extract_with_database()  # bu fonksiyonun içerisine query yerleştirilecek

    """
    Kullanıcının seçeceği featureleri, belki önceden seçtikleri burada seçtirelim.
    """

    if use_feature_explorer:
        FExplorer = FeatureExplorer(path_dict)

        features = ['mfcc']
        title = 'new'
        note = 'en son cıkarttıgım featureler.'
        FExplorer.save_from_temp(features, title, note)  # eğer feature extraction yapıldıysa

        # List all features

    if build_your_model:
        conv_1d = {'type': 'conv_1d', 'filters': 32, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'}
        dropout = {'type': 'dropout', 'rate': 0.5}
        dense = {'type': 'dense', 'units': 32, 'activation': 'relu'}
        batch_normalization = {'type': 'batch_normalization'}
        flatten = {'type': 'flatten'}
        dense_2 = {'type': 'dense', 'units': 32, 'activation': 'relu'}
        compile_config = {'optimizer': 'rmsprop', 'loss': 'sparse_categorical_crossentropy', 'metrics': ['accuracy']}

        my_layers = [conv_1d, flatten, dropout, dense, dense_2]

        model_builder = NewModelBuilder(path_dict, my_layers, compile_config)
        model_builder.build()

    if train_your_model:
        model_train_config = {'save_model': True, 'test_split_rate': 0.30, 'batch_size': 2, 'epochs': 50,
                              'validation_split_rate': 0.2, 'use_random_state': True}

        tensorboard_config = {'use_tensorboard': True}

        model_trainer = ModelTrainer(model_train_config=model_train_config, path_dict=path_dict,
                                     tensorboard_config=tensorboard_config)
        model_trainer.train_with_temp_features()
        #model_trainer.train_with_temp_features(compiled_model)


if __name__ == "__main__":
    main()
