from flask_sqlalchemy import SQLAlchemy
from app import db


class DatasetCatalog(db.Model):
    __tablename__ = 'dataset_catalog'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True)
    isdownloaded = db.Column(db.String(1))
    ismeta = db.Column(db.String(1))
    children = db.relationship("DatasetMeta", backref="dataset_catalog" ,lazy=True)

    def __init__(self, name, isdownloaded, ismeta):
        self.name = name
        self.isdownloaded = isdownloaded
        self.ismeta = ismeta

    def __repr__(self):
        return '<DatasetCatalog %r>' % (self.id)


class DatasetMeta(db.Model):
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
