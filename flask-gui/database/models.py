from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class DbDatasetCatalog(db.Model):
    __tablename__ = 'dataset_catalog'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True)
    isdownloaded = db.Column(db.String(1))
    ismeta = db.Column(db.String(1))
    children = db.relationship("DbDatasetMeta", backref="dataset_catalog", lazy=True)

    def __init__(self, name, isdownloaded, ismeta):
        self.name = name
        self.isdownloaded = isdownloaded
        self.ismeta = ismeta

    def __repr__(self):
        return '<DatasetCatalog %r>' % self.id


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
    DOWNLOADS_FOLDER = db.Column(db.String(500))
    DATASETS_FOLDER = db.Column(db.String(500))
    RAVDESS_FILES_PATH = db.Column(db.String(500))
    CREMA_D_FILES_PATH = db.Column(db.String(500))
    SAVEE_FILES_PATH = db.Column(db.String(500))
    EMODB_FILES_PATH = db.Column(db.String(500))
    DATA_METADATA_DF_PATH = db.Column(db.String(500))

    def __init__(self, TRAINING_FILES_PATH, TRAINING_FILES_SPECTOGRAM, SAVE_RUNTIME_FEATURES, SAVE_RUNTIME_FEATURES_X,
                 SAVE_RUNTIME_FEATURES_Y, MODEL_FEATURES_PATH, MODEL_WEIGHTS_PATH, MODEL_TRAINING_PLOTS,
                 TEST_FILES_PATH, RAVDESS_FILES_PATH, CREMA_D_FILES_PATH, SAVEE_FILES_PATH, EMODB_FILES_PATH,
                 DATA_METADATA_DF_PATH, DOWNLOADS_FOLDER, DATASETS_FOLDER):
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
        self.DOWNLOADS_FOLDER = DOWNLOADS_FOLDER
        self.DATASETS_FOLDER = DATASETS_FOLDER

    def __repr__(self):
        return '<DBCONFIG %r>' % (self.id)


class DbModel(db.Model):
    __tablename__ = 'model'
    id = db.Column(db.Integer, primary_key=True)
    type = db.Column(db.String(500))
    filters = db.Column(db.String(500))
    kernel_size = db.Column(db.String(500))
    padding = db.Column(db.String(500))
    activation = db.Column(db.String(500))
    rate = db.Column(db.String(500))
    units = db.Column(db.String(500))
    optimizer = db.Column(db.String(500))
    loss = db.Column(db.String(500))
    metrics = db.Column(db.String(500))

    def __init__(self, type, filters="", kernel_size="", padding="", activation="", rate="", units="", optimizer="",
                 loss="", metrics=""):
        self.type = type
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.rate = rate
        self.units = units
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    def __repr__(self):
        return '<DBCONFIG %r>' % (self.id)
