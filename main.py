def main():
    # deneme
    from FeatureExtractor import FeatureExtractor
    from DatasetExplorer import DatasetExplorer
    import numpy as np
    from DataMetaDataCreator import MetaDataCreator
    from NewModelBuilder import NewModelBuilder
    from ModelTrainer import ModelTrainer

    explore_datasets = False
    make_feature_extraction = True
    create_meta_csv = False
    select_pretrained_model = False
    build_your_model = False
    train_your_model = False

    """initializing neccesary vars"""
    path_dict = {'DOWNLOADS_FOLDER': 'Downloads', 'DATASETS_FOLDER': 'Datasets',
                 'emoDB': 'Datasets/emoDB', 'Ravdess': 'Datasets/Ravdess',
                 'SAVEE': 'Datasets/SAVEE', 'Crema-D': 'Datasets/Crema-D',
                 'TEMP_FOLDER': 'TEMP'}

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
        feature_extraction_dict = {'sampling_rate': 44100, 'samples': 44100 * 4, 'n_mfcc': 40,
                                   'features': ['mfcc'],
                                   'augmentations': ['white_noise', 'stretch', 'shift', 'change_speed']}
        f = FeatureExtractor(feature_extraction_dict)  # feature_extraction_dict {} yolla,

        f.extract_with_database() # bu fonksiyonun içerisine query yerleştirilecek

    if build_your_model:
        model_builder = NewModelBuilder()

        input_layer = {'name': 'input_layer', 'input_shape': (40, 1), 'batch_size': (40)}
        conv_1d = {'name': 'conv_1d', 'filters': 32, 'kernel_size': 3, 'padding': 'same', 'activation':'relu'}
        dropout = {'name': 'dropout', 'rate': 0.5}
        dense = {'name': 'dense', 'units': 32, 'activation': 'relu'}
        batch_normalization = {'name': 'batch_normalization'}
        flatten = {'name':'flatten'}
        compile_config = {'optimizer': 'rmsprop', 'loss': 'binary_crossentropy', 'metrics': ['accuracy']}

        my_layers = [input_layer, conv_1d, dropout, dense, batch_normalization,flatten, dense]

        uncompiled_model = model_builder.get_uncompiled_model(my_layers)
        compiled_model = model_builder.get_compiled_model(compile_config)

    if train_your_model:
        model_train_config = {'save_model':True, 'split_rate':0.66}
        model_trainer = ModelTrainer(model_train_config,path_dict)
        model_trainer.train_with_temp_features()









if __name__ == "__main__":
    main()
