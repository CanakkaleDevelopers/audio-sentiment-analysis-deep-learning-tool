from flask_sqlalchemy import SQLAlchemy
from app import db


class DbDatasetCatalog(db.Model):
    __tablename__ = 'dataset_catalog'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True)
    isdownloaded = db.Column(db.String(1))
    ismeta = db.Column(db.String(1))
    children = db.relationship("DbDatasetMeta", backref="dataset_catalog" ,lazy=True)

    def __init__(self, name, isdownloaded, ismeta):
        self.name = name
        self.isdownloaded = isdownloaded
        self.ismeta = ismeta

    def __repr__(self):
        return '<DatasetCatalog %r>' % (self.id)


class DbDatasetMeta(db.Model):
    __tablename__ = 'dataset_meta'
    id = db.Column(db.Integer, primary_key=True)
    dataset_catalog_id = db.Column(db.Integer, db.ForeignKey('dataset_catalog.id'))
    gender = db.Column(db.String(255))
    emotion = db.Column(db.String(255))
    path = db.Column(db.String(500))

    def __init__(self, dataset_catalog_id, gender, emotion, path):
        self.dataset_catalog_id = dataset_catalog_id
        self.gender = gender
        self.emotion = emotion
        self.path = path

    def __repr__(self):
        return '<DatasetMeta %r>' % (self.id)


class DbConfig(db.Model):
    __tablename__ = 'config'
    id = db.Column(db.Integer, primary_key=True)
    TRAINING_FILES_PATH = db.Column(db.String(500))
    TRAINING_FILES_SPECTOGRAM = db.Column(db.String(500))
    SAVE_RUNTIME_FEATURES = db.Column(db.String(500))
    SAVE_RUNTIME_FEATURES_X = db.Column(db.String(500))
    SAVE_RUNTIME_FEATURES_Y = db.Column(db.String(500))
    MODEL_FEATURES_PATH = db.Column(db.String(500))
    MODEL_WEIGHTS_PATH = db.Column(db.String(500))
    MODEL_TRAINING_PLOTS = db.Column(db.String(500))
    TEST_FILES_PATH = db.Column(db.String(500))
    RAVDESS_FILES_PATH = db.Column(db.String(500))
    CREMA_D_FILES_PATH = db.Column(db.String(500))
    SAVEE_FILES_PATH = db.Column(db.String(500))
    EMODB_FILES_PATH = db.Column(db.String(500))
    DATA_METADATA_DF_PATH = db.Column(db.String(500))

    def __init__(self, TRAINING_FILES_PATH, TRAINING_FILES_SPECTOGRAM, SAVE_RUNTIME_FEATURES,SAVE_RUNTIME_FEATURES_X,SAVE_RUNTIME_FEATURES_Y,MODEL_FEATURES_PATH,MODEL_WEIGHTS_PATH,MODEL_TRAINING_PLOTS,TEST_FILES_PATH,RAVDESS_FILES_PATH,CREMA_D_FILES_PATH,SAVEE_FILES_PATH,EMODB_FILES_PATH,DATA_METADATA_DF_PATH):
        self.TRAINING_FILES_PATH = TRAINING_FILES_PATH
        self.TRAINING_FILES_SPECTOGRAM = TRAINING_FILES_SPECTOGRAM
        self.SAVE_RUNTIME_FEATURES = SAVE_RUNTIME_FEATURES
        self.SAVE_RUNTIME_FEATURES_X = SAVE_RUNTIME_FEATURES_X
        self.SAVE_RUNTIME_FEATURES_Y = SAVE_RUNTIME_FEATURES_Y
        self.MODEL_FEATURES_PATH = MODEL_FEATURES_PATH
        self.MODEL_WEIGHTS_PATH = MODEL_WEIGHTS_PATH
        self.MODEL_TRAINING_PLOTS = MODEL_TRAINING_PLOTS
        self.TEST_FILES_PATH = TEST_FILES_PATH
        self.RAVDESS_FILES_PATH = RAVDESS_FILES_PATH
        self.CREMA_D_FILES_PATH = CREMA_D_FILES_PATH
        self.SAVEE_FILES_PATH = SAVEE_FILES_PATH
        self.EMODB_FILES_PATH = EMODB_FILES_PATH
        self.DATA_METADATA_DF_PATH = DATA_METADATA_DF_PATH

    def __repr__(self):
        return '<DatasetMeta %r>' % (self.id)
