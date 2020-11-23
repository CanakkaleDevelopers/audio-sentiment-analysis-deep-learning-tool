import testmultithread
from flask import Flask, render_template

from flask_sqlalchemy import SQLAlchemy

from multiprocessing import Process

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database/data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
db = SQLAlchemy(app)

from database.models import DbDatasetCatalog, DbDatasetMeta, DbConfig

#veritabanı modellerden bilgilerini alıp oluşturuluyor.
db.metadata.create_all(db.engine)

@app.route('/')
def page_welcome():
    init_config()
    return render_template("index.html")


@app.route('/test')
def temp_func():
    testxd = DbDatasetCatalog.query.first().children
    for a in testxd:
        print(a.gender)

    return render_template("index.html")


@app.teardown_appcontext
def shutdown_session(exception=None):
    db.session.remove()


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
            TRAINING_FILES_PATH = str(working_dir_path) + '\\pass\\'
            TRAINING_FILES_SPECTOGRAMS = str(working_dir_path) + '\\TEMP\\Spectogram\\'
            SAVE_RUNTIME_FEATURES = str(working_dir_path) + '\\TEMP\\'
            SAVE_RUNTIME_FEATURES_X = str(working_dir_path) + '\\TEMP\\featuresX.npy'
            SAVE_RUNTIME_FEATURES_Y = str(working_dir_path) + '\\TEMP\\featuresY.npy'
            MODEL_FEATURES_PATH = str(working_dir_path) + '\\ImplementedModels\\'
            MODEL_WEIGHTS_PATH = str(working_dir_path) + '\\ModelWeights\\'
            MODEL_TRAINING_PLOTS = str(working_dir_path) + '\\TEMP\\Plots'
            TEST_FILES_PATH = str(working_dir_path) + '\\pass\\'
            RAVDESS_FILES_PATH = str(working_dir_path) + '\\Datasets\\Ravdess'
            CREMA_D_FILES_PATH = str(working_dir_path) + '\\Datasets\\Crema-D'
            SAVEE_FILES_PATH = str(working_dir_path) + '\\Datasets\\SAVEE'
            EMODB_FILES_PATH = str(working_dir_path) + '\\Datasets\\emoDB'
            DATA_METADATA_DF_PATH = str(working_dir_path) + '\\TEMP\\datatable.csv'
            db_config = DbConfig(TRAINING_FILES_PATH, TRAINING_FILES_SPECTOGRAMS, SAVE_RUNTIME_FEATURES,
                                 SAVE_RUNTIME_FEATURES_X, SAVE_RUNTIME_FEATURES_Y, MODEL_FEATURES_PATH,
                                 MODEL_WEIGHTS_PATH, MODEL_TRAINING_PLOTS, TEST_FILES_PATH, RAVDESS_FILES_PATH,
                                 CREMA_D_FILES_PATH, SAVEE_FILES_PATH, EMODB_FILES_PATH, DATA_METADATA_DF_PATH)
            db.session.add(db_config)
            db.session.flush()

        else:
            TRAINING_FILES_PATH = str(working_dir_path) + '/pass/'
            TRAINING_FILES_SPECTOGRAMS = str(working_dir_path) + '/TEMP/Spectogram'
            MODEL_DIR_PATH = str(working_dir_path) + '/ImplementedModels/'
            MODEL_WEIGHTS_PATH = str(working_dir_path) + '/ModelWeights'
            MODEL_TRAINING_PLOTS = str(working_dir_path) + 'TEMP/Plots'
            SAVE_RUNTIME_FEATURES = str(working_dir_path) + '/TEMP/'
            SAVE_RUNTIME_FEATURES_X = str(working_dir_path) + '/TEMP/featuresX.npy'
            SAVE_RUNTIME_FEATURES_Y = str(working_dir_path) + '/TEMP/featuresY.npy'
            MODEL_FEATURES_PATH = str(working_dir_path) + '/ImplementedModels/'
            TEST_FILES_PATH = str(working_dir_path) + '/pass/'
            RAVDESS_FILES_PATH = str(working_dir_path) + '/Datasets/Ravdess'
            CREMA_D_FILES_PATH = str(working_dir_path) + '/Datasets/Crema-D'
            SAVEE_FILES_PATH = str(working_dir_path) + '/Datasets/SAVEE'
            EMODB_FILES_PATH = str(working_dir_path) + '/Datasets/emoDB'
            DATA_METADATA_DF_PATH = str(working_dir_path) + '/TEMP/metadata_table.csv'
            db_config = DbConfig(TRAINING_FILES_PATH, TRAINING_FILES_SPECTOGRAMS, SAVE_RUNTIME_FEATURES,
                                 SAVE_RUNTIME_FEATURES_X, SAVE_RUNTIME_FEATURES_Y, MODEL_FEATURES_PATH,
                                 MODEL_WEIGHTS_PATH, MODEL_TRAINING_PLOTS, TEST_FILES_PATH, RAVDESS_FILES_PATH,
                                 CREMA_D_FILES_PATH, SAVEE_FILES_PATH, EMODB_FILES_PATH, DATA_METADATA_DF_PATH)
            db.session.add(db_config)
            init_DbDatasetCatalog()
            db.session.commit()

    else:
        print("Config Dosyası bulundu.")

    print("Config İşlemi Tamamlandı.")

    print("----------------")


def init_DbDatasetCatalog():
    print("----------------")
    print("Catalog işlemi başlatılıyor...")
    DbDatasetCatalog.query.delete()
    db.session.add(DbDatasetCatalog("Crema-D",0,0))
    db.session.add(DbDatasetCatalog("emoDB",0,0))
    db.session.add(DbDatasetCatalog("Ravdess",0,0))
    db.session.add(DbDatasetCatalog("SAVEE",0,0))
    print("Catalog İşlemi Tamamlandı.")
    print("----------------")


if __name__ == '__main__':
    app.run(threaded=True)
