def main():
    # deneme
    from FeatureExtractor import FeatureExtractor
    from DatasetExplorer import DatasetExplorer
    import numpy as np
    from DataMetaDataCreator import MetaDataCreator

    explore_datasets = False
    make_feature_extraction = False
    create_meta_csv = False
    select_pretrained_model = False
    build_your_model = True

    """initializing neccesary vars"""
    path_dict = {'downloads_folder': 'Downloads', 'datasets_folder': 'Datasets',
                 'emoDB': 'Datasets/emoDB', 'Ravdess': 'Datasets/Ravdess',
                 'SAVEE': 'Datasets/SAVEE', 'Crema-D': 'Datasets/Crema-D',
                 'temp_folder': 'TEMP'}

    """0.5 inci faz veriseti tarama ve indirme"""
    if explore_datasets:
        demanded_datasets = ['Ravdess']
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
                                   'features': ['mfcc', 'chroma', 'mel', 'tonnetz', 'mfcc_delta'],
                                   'augmentations': ['white_noise', 'stretch', 'shift', 'change_speed']}
        f = FeatureExtractor(feature_extraction_dict)  # feature_extraction_dict {} yolla,

        # Before extraction loop variables decleration block
        save_features_X = 'TEMP/FeaturesX'  # temp_featuresX doya yolu
        save_features_Y = 'TEMP/FeaturesY'  # temp_featuresY dosya yolu
        _, features_x_lenght = f.extract('example_audio.ogg')  # dummy # burası değiştirilmememli
        features_x = np.empty(features_x_lenght)
        features_y = []

        # Looping all records block
        lenght_of_records_in_database = 1000  # kayıtların toplam sayısı
        for count in range(lenght_of_records_in_database):  # -> veritabanındaki kayıt sayısı kadar dön

            # query record block
            record = {'Gender': 'Male', 'Emotion': 'Happy', 'Source': 'Ravdess', 'Path': 'example_audio.ogg',
                      'augment': True}  # burayı queryi yap sırayla oku qureyi yi bu şekilde getir veya alt tarafları düzenle

            print('Extracting selected features from  {} {} {} audio record. {} file left'.format(record['Source'],
                                                                                                  record['Emotion'],
                                                                                                  record['Gender'], (
                                                                                                          lenght_of_records_in_database - count)))

            # Extracting file himself
            record_features, _ = f.extract(record['Path'])
            record_label = record['Emotion']

            features_x = np.vstack([features_x, record_features])
            features_y = np.hstack([features_y, record_label])

            # Extracting augmented file if True
            if record['augment']:
                print('Extracting AUGMENTED record features from  {} {} {} audio record.'.format(
                    record['Source'],
                    record['Emotion'],
                    record['Gender'], ))
                record_features, _ = f.extract(record['Path'], make_augmentations=True)
                record_label = record['Emotion']

                features_x = np.vstack([features_x, record_features])
                features_y = np.hstack([features_y, record_label])

            # save block

            features_x = features_x[1:]  # trim first np.empty(40)
            np.save(save_features_X, features_x)
            np.save(save_features_Y, features_y)

            """
            if(save_this_features_forever):
                TODO-> eğer kullanıcı bu featureleri sonra da kullanmak isterse kaydedebilmeli
                çünkü işlem çok uzun
            """

    if build_your_model:
        pass


if __name__ == "__main__":
    main()
