def main():
    # deneme
    from FeatureExtractor import FeatureExtractor
    from DataMetaDataCreator import MetaDataCreator

    # 1. Faz verisetlerinin meta çıkartımı TODO-> fonksiyon haline getir TODO-> daha çok veriseti ekle
    MetaDataCreator.ravdess_to_datatable()
    MetaDataCreator.cremad_to_datatable()
    MetaDataCreator.savee_to_datatable()
    MetaDataCreator.emodb_to_datatable()

    """
    1.5 Faz  -> 1. fazın yarattığı metadata_table.csv dosyasından kullanıcının isteklerine göre istenmeyen verileri sil,
    ve dataframei karıştır, tekrar aynı isimle ve aynı yere metadata_table.csv dosyasını kaydet.
    """

    """
    2. Faz -> Öznitelikleri çıkart ve TEMP/FeaturesX.npy ve TEMP/FeaturesY.npy dosyaları oluşsun.
    """
    #FeatureExtractor.extract_from_metadata_table()





if __name__ == "__main__":
    main()
