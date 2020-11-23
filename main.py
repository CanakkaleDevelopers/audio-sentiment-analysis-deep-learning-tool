def main():
    # deneme
    from FeatureExtractor import FeatureExtractor
    import numpy as np
    from DataMetaDataCreator import MetaDataCreator

    # 1. Faz verisetlerinin meta çıkartımı TODO-> fonksiyon haline getir TODO-> daha çok veriseti ekle
    # MetaDataCreator.ravdess_to_datatable()
    # MetaDataCreator.cremad_to_datatable()
    # MetaDataCreator.savee_to_datatable()
    # MetaDataCreator.emodb_to_datatable()

    """
    1.5 Faz  -> 1. fazın yarattığı metadata_table.csv dosyasından kullanıcının isteklerine göre istenmeyen verileri sil,
    ve dataframei karıştır, tekrar aynı isimle ve aynı yere metadata_table.csv dosyasını kaydet.
    """

    """
    2. Faz -> Öznitelikleri çıkart ve TEMP/FeaturesX.npy ve TEMP/FeaturesY.npy dosyaları oluşsun.
    """

    ## initialization of FeatureExtractor class block
    feature_extraction_dict = {'sampling_rate': 44100, 'samples': 44100 * 4, 'n_mfcc': 40,
                               'features': ['mfcc', 'chroma', 'mel', 'tonnetz', 'mfcc_delta'],
                               'augmentations': ['white_noise', 'stretch', 'shift', 'change_speed']}
    f = FeatureExtractor(feature_extraction_dict)

    # Before extraction loop variables decleration block
    save_features_X = 'TEMP/FeaturesX'
    save_features_Y = 'TEMP/FeaturesY'
    _, features_x_lenght = f.extract('example_audio.ogg')  # dummy # burası değiştirilmememli
    features_x = np.empty(features_x_lenght)
    features_y = []

    # Looping all records block
    lenght_of_records_in_database = 1000
    for count in range(lenght_of_records_in_database):  # -> veritabanındaki kayıt sayısı kadar dön

        # query record block
        record = {'Gender': 'Male', 'Emotion': 'Happy', 'Source': 'Ravdess', 'Path': 'example_audio.ogg',
                  'augment': True}

        print('Extracting selected features from  {} {} {} audio record. {} file left'.format(record['Source'],
                                                                                              record['Emotion'],
                                                                                              record['Gender'], (
                                                                                                      lenght_of_records_in_database - count)))

        # Extracting file himself
        record_features, _  = f.extract(record['Path'])
        record_label = record['Emotion']

        features_x = np.vstack([features_x, record_features])
        features_y = np.hstack([features_y, record_label])

        # Extracting augmented file if True
        if record['augment']:
            print('Extracting AUGMENTED record features from  {} {} {} audio record.'.format(
                record['Source'],
                record['Emotion'],
                record['Gender'],))
            record_features, _ = f.extract(record['Path'], make_augmentations=True)
            record_label = record['Emotion']

            features_x = np.vstack([features_x, record_features])
            features_y = np.hstack([features_y, record_label])

        # save block

        features_x = features_x[1:]  # trim first np.empty(40)
        np.save(save_features_X, features_x)
        np.save(save_features_Y, features_y)


if __name__ == "__main__":
    main()
