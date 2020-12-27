from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from library.DatasetExplorer import DatasetExplorer
from library.DataMetaDataCreator import MetaDataCreator
from library.FeatureExtractor import FeatureExtractor
from library.ModelBuilder import NewModelBuilder
from library.ModelTrainer import ModelTrainer
from database.models import DbDatasetCatalog, DbDatasetMeta, DbConfig, db, DbModel

from multiprocessing import Process

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database/data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
db.init_app(app)


@app.route('/')
def page_welcome():
    init_config()
    db.session.commit()
    return render_template("index.html")


@app.route('/select_dataset')
def web_select_dataset():
    return render_template("select_dataset.html", datasets=DbDatasetCatalog.query.all())


@app.route('/download_dataset', methods=['POST'])
def web_download_dataset():
    download_datasets(list(request.form.keys()))
    return "download_dataset_finished"


@app.route('/select_metadata')
def web_select_metadata():
    return render_template("select_metadata.html", datasets=DbDatasetCatalog.query.all())


@app.route('/create_metadata', methods=['POST'])
def web_create_metadata():
    metadata_creator = MetaDataCreator(DbConfig.query.first().__dict__)
    metadata_creator.create_csv(list(request.form.keys()))
    datametacsv_to_database()
    return "tamam"


@app.route("/select_features")
def web_select_features():
    return render_template("select_features.html")


@app.route("/create_features", methods=['POST'])
def web_create_features():
    from distutils.util import strtobool
    # distutils.util.strtobool()
    # feature_extraction_dict = {'sampling_rate': 44100, 'duration': 4, 'trim_long_data': False, 'n_mfcc': 40,
    #                            'features': ['mfcc'],
    #                            'augmentations': ['white_noise', 'stretch', 'shift', 'change_speed']}
    # data_augmentation_dict = {'shift_rate': 1600, 'stretch_rate': 1, 'speed_change': 1,
    #                           'pitch_pm': 24, 'bins_per_octave': 24}
    # f = FeatureExtractor(feature_extraction_dict, data_augmentation_dict)  # feature_extraction_dict {} yolla,
    #
    # f.extract_with_database()  # bu fonksiyonun içerisine query yerleştirilecek

    # Bu işlem selectbox üzerinden gelen verilerin flatten dict üzerinde gösterilememesinden dolayı yazılmıştır.
    a = request.form.to_dict(flat=True)
    a['features'] = request.form.getlist("features")
    a['augmentations'] = request.form.getlist("augmentations")
    a['sampling_rate'] = int(a['sampling_rate'])
    a['duration'] = int(a['duration'])
    a['n_mfcc'] = int(a['n_mfcc'])
    a['pitch_pm'] = int(a['pitch_pm'])
    a['bins_per_octave'] = int(a['bins_per_octave'])
    a['shift_rate'] = int(a['shift_rate'])
    a['speed_change'] = float(a['speed_change'])
    a['trim_long_data'] = bool(strtobool(a['trim_long_data']))

    print(a)
    f = FeatureExtractor(a, a, DbDatasetMeta.query.all())  # feature_extraction_dict {} yolla

    f.extract_with_database()

    return "finished"


@app.route('/create_model')
def web_create_model():
    return render_template("create_model.html", columns=DbModel.__table__.columns.keys(),
                           layers=DbModel.query.all())


@app.route('/create_model_conv_1d', methods=['POST', 'GET'])
def web_create_model_conv_1d():
    if request.method == 'POST':
        model_conv1d = DbModel(type="conv_1d", filters=request.form.get("filters"),
                               kernel_size=request.form.get("kernel_size"), padding=request.form.get("padding"))
        db.session.add(model_conv1d)
        db.session.flush()
        db.session.commit()
        return redirect(url_for('web_create_model'))
        # print(request.form.to_dict(flat=True))
    else:
        return render_template("create_model_conv_1d.html")


@app.route('/create_model_dropout', methods=['POST', 'GET'])
def web_create_model_dropout():
    if request.method == 'POST':
        model_dropout = DbModel(type="dropout", rate=request.form.get("rate"))
        db.session.add(model_dropout)
        db.session.flush()
        db.session.commit()
        return redirect(url_for('web_create_model'))
        # print(request.form.to_dict(flat=True))
    else:
        return render_template("create_model_dropout.html")


@app.route('/create_model_dense', methods=['POST', 'GET'])
def web_create_model_dense():
    if request.method == 'POST':
        model_dense = DbModel(type="dense", units=request.form.get("units"), activation=request.form.get("activation"))
        db.session.add(model_dense)
        db.session.flush()
        db.session.commit()
        return redirect(url_for('web_create_model'))
        # print(request.form.to_dict(flat=True))
    else:
        return render_template("create_model_dense.html")


@app.route('/create_model_batch_normalization')
def web_create_model_batch_normalization():
    model_batch = DbModel(type="batch_normalization")
    db.session.add(model_batch)
    db.session.flush()
    db.session.commit()
    return redirect(url_for('web_create_model'))


@app.route('/create_model_flatten')
def web_create_model_flatten():
    model_flatten = DbModel(type="flatten")
    db.session.add(model_flatten)
    db.session.flush()
    db.session.commit()
    return redirect(url_for('web_create_model'))


@app.route('/delete_layer/<layer_id>')
def web_delete_model_layer(layer_id):
    db.session.delete(DbModel.query.get(layer_id))
    db.session.commit()
    return redirect(url_for('web_create_model'))


@app.route("/select_compile_config")
def web_select_compile_config():
    return render_template("select_compile_config.html")


@app.route("/create_compile_config", methods=['POST'])
def web_create_compile_config():
    a = request.form.to_dict(flat=True)
    model_builder = NewModelBuilder(DbConfig.query.first().__dict__, DbModel.query.all(), a)
    model_builder.build()
    db.session.commit()
    return "finished"\

@app.route("/select_model_trainer")
def web_select_model_trainer():
    return render_template("select_model_trainer.html")


@app.route("/model_trainer", methods=['POST'])
def web_model_trainer():
    from distutils.util import strtobool
    a = request.form.to_dict(flat=True)
    a['save_model'] = bool(strtobool(a['save_model']))
    a['use_random_state'] = bool(strtobool(a['use_random_state']))
    a['use_tensorboard'] = bool(strtobool(a['use_tensorboard']))

    a['test_split_rate'] = float(a['test_split_rate'])
    a['batch_size'] = int(a['batch_size'])
    a['epochs'] = int(a['epochs'])
    a['validation_split_rate'] = float(a['validation_split_rate'])
    model_trainer = ModelTrainer(model_train_config=a, path_dict=DbConfig.query.first().__dict__,
                                    tensorboard_config=a)
    model_trainer.train_with_temp_features()
    db.session.commit()



    print(a)
    # model_builder = NewModelBuilder(DbConfig.query.first().__dict__, DbModel.query.all(), a)
    # model_builder.build()
    return "finished"


@app.route('/test')
def temp_func():
    print()
    return "ok"


@app.teardown_appcontext
def shutdown_session(exception=None):
    db.session.remove()


"""
1-Program veritabanı dosyasını kontrol eder.
2-Program veritabanı üzerinde Config tablosu üzerinde aramalar yapar ve bozulma var ise alttaki fonksiyonlar çalışır.
"""


# Config olarak kullanılan kesinleştirilmiş veri yolları veri tabanına eğer kayıt yok ise kaydediliyor.
# Bu kayıtlar buradan değiştirilip veritabanı silindiği zaman otomatik olarak değiştirilmiş olarak kaydedilecektir.
def init_config():
    print("----------------")
    print("Config işlemi başlatılıyor...")

    if (DbConfig.query.all().__len__() == 0):
        print("Config Bulunamadı , Oluşturuluyor.")
        import os
        import sys
        import pathlib

        working_dir_path = os.path.dirname(os.path.abspath(__file__))
        if sys.platform.startswith('win32'):
            # DATASET INDIRME KISMI
            DOWNLOADS_FOLDER = 'Downloads\\'
            DATASETS_FOLDER = 'Datasets\\'
            RAVDESS_FILES_PATH = 'Datasets\\Ravdess\\'
            CREMA_D_FILES_PATH = 'Datasets\\Crema-D\\'
            SAVEE_FILES_PATH = 'Datasets\\SAVEE\\'
            EMODB_FILES_PATH = 'Datasets\\emoDB\\'

            TRAINING_FILES_PATH = 'pass\\'
            TRAINING_FILES_SPECTOGRAMS = 'TEMP\\Spectogram\\'
            SAVE_RUNTIME_FEATURES = 'TEMP\\'
            SAVE_RUNTIME_FEATURES_X = 'TEMP\\featuresX.npy'
            SAVE_RUNTIME_FEATURES_Y = 'TEMP\\featuresY.npy'
            MODEL_FEATURES_PATH = 'ImplementedModels\\'
            MODEL_WEIGHTS_PATH = 'ModelWeights\\'
            MODEL_TRAINING_PLOTS = 'TEMP\\Plots\\'
            TEST_FILES_PATH = 'pass\\'
            DATA_METADATA_DF_PATH = 'TEMP\\datatable.csv'
            db_config = DbConfig(TRAINING_FILES_PATH, TRAINING_FILES_SPECTOGRAMS, SAVE_RUNTIME_FEATURES,
                                 SAVE_RUNTIME_FEATURES_X, SAVE_RUNTIME_FEATURES_Y, MODEL_FEATURES_PATH,
                                 MODEL_WEIGHTS_PATH, MODEL_TRAINING_PLOTS, TEST_FILES_PATH, RAVDESS_FILES_PATH,
                                 CREMA_D_FILES_PATH, SAVEE_FILES_PATH, EMODB_FILES_PATH, DATA_METADATA_DF_PATH,
                                 DOWNLOADS_FOLDER, DATASETS_FOLDER)
            db.session.add(db_config)
            db.session.flush()

        else:
            # DATASET INDIRME KISMI
            DOWNLOADS_FOLDER = 'Downloads/'
            RAVDESS_FILES_PATH = 'Datasets/Ravdess/'
            CREMA_D_FILES_PATH = 'Datasets/Crema-D/'
            SAVEE_FILES_PATH = 'Datasets/SAVEE/'
            EMODB_FILES_PATH = 'Datasets/emoDB/'
            DATASETS_FOLDER = 'Datasets/'

            TRAINING_FILES_PATH = 'pass/'
            TRAINING_FILES_SPECTOGRAMS = 'TEMP/Spectogram/'
            MODEL_DIR_PATH = 'ImplementedModels/'
            MODEL_WEIGHTS_PATH = 'ModelWeights/'
            MODEL_TRAINING_PLOTS = 'TEMP/Plots/'
            SAVE_RUNTIME_FEATURES = 'TEMP/'
            SAVE_RUNTIME_FEATURES_X = 'TEMP/featuresX.npy'
            SAVE_RUNTIME_FEATURES_Y = 'TEMP/featuresY.npy'
            MODEL_FEATURES_PATH = 'ImplementedModels/'
            TEST_FILES_PATH = 'pass/'
            DATA_METADATA_DF_PATH = 'TEMP/metadata_table.csv'
            db_config = DbConfig(TRAINING_FILES_PATH, TRAINING_FILES_SPECTOGRAMS, SAVE_RUNTIME_FEATURES,
                                 SAVE_RUNTIME_FEATURES_X, SAVE_RUNTIME_FEATURES_Y, MODEL_FEATURES_PATH,
                                 MODEL_WEIGHTS_PATH, MODEL_TRAINING_PLOTS, TEST_FILES_PATH, RAVDESS_FILES_PATH,
                                 CREMA_D_FILES_PATH, SAVEE_FILES_PATH, EMODB_FILES_PATH, DATA_METADATA_DF_PATH,
                                 DOWNLOADS_FOLDER, DATASETS_FOLDER)
            db.session.add(db_config)
            init_DbDatasetCatalog()
            db.session.commit()

    else:
        print("Config Dosyası bulundu.")

    print("Config İşlemi Tamamlandı.")

    print("----------------")


# Şu anda analiz edilip kullanılması için dosya tanımlama algoritmaları yazılmış veri setleri tanımlamalarıdır.
# Yeni bir veri seti algoritması eklendiği zaman ortak karar ile buraya eklemeler yapılabilir.
# Veritabanı dosyası silindiği taktirde kendisini güncel verilerle oluşturacaktır.
def init_DbDatasetCatalog():
    print("----------------")
    print("Catalog işlemi başlatılıyor...")
    DbDatasetCatalog.query.delete()
    db.session.add(DbDatasetCatalog("Crema-D", 0, 0))
    db.session.add(DbDatasetCatalog("emoDB", 0, 0))
    db.session.add(DbDatasetCatalog("Ravdess", 0, 0))
    db.session.add(DbDatasetCatalog("SAVEE", 0, 0))
    print("Catalog İşlemi Tamamlandı.")
    print("----------------")


def download_datasets(datasets):
    print("----------------")
    print("Dataset indirme işlemi başlatılıyor...")
    # print(DbConfig.query.first().__dict__)
    dataset_explorer = DatasetExplorer(datasets, DbConfig.query.first().__dict__)
    dataset_explorer.scan()
    dataset_explorer.download_datasets()
    print("Dataset indirme işlemi tamamlandı.")
    print("----------------")
    del dataset_explorer


def datametacsv_to_database():
    DbDatasetMeta.query.delete()
    db.session.commit()
    import pandas as pd
    metadata_df = pd.read_csv(DbConfig.query.first().DATA_METADATA_DF_PATH)
    metadata_df = metadata_df.loc[:, ~metadata_df.columns.str.contains('^Unnamed')]
    metadatas = []
    for index, row in metadata_df.iterrows():
        dataset = DbDatasetCatalog.query.filter(DbDatasetCatalog.name == row['source']).first()
        metadatas.append(DbDatasetMeta(path=row['path'], gender=row["gender"], emotion=row["emotion"],
                                       dataset_catalog_id=dataset.id))
    print(metadatas.__len__())
    db.session.bulk_save_objects(metadatas)
    db.session.commit()
    from os import remove
    remove(DbConfig.query.first().DATA_METADATA_DF_PATH)


if __name__ == '__main__':
    db.app = app
    db.create_all()
    app.run(threaded=True, debug=True)
