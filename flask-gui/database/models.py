from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from database.database import Base


class DatasetCatalog(Base):
    __tablename__ = 'dataset_catalog'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True)
    isdownloaded = Column(String(1))
    ismeta = Column(String(1))
    children = relationship("DatasetMeta", backref="dataset_catalog")

    def __init__(self, name, isdownloaded, ismeta):
        self.name = name
        self.isdownloaded = isdownloaded
        self.ismeta = ismeta

    def __repr__(self):
        return '<DatasetCatalog %r>' % (self.id)


class DatasetMeta(Base):
    __tablename__ = 'dataset_meta'
    id = Column(Integer, primary_key=True)
    dataset_catalog_id = Column(Integer, ForeignKey('dataset_catalog.id'))
    gender = Column(String(255))
    emotion = Column(String(255))
    path = Column(String(500))

    def __init__(self, dataset_catalog_id, gender, emotion, path):
        self.dataset_catalog_id = dataset_catalog_id
        self.gender = gender
        self.emotion = emotion
        self.path = path

    def __repr__(self):
        return '<DatasetMeta %r>' % (self.id)
