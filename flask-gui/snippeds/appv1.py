import testmultithread
from flask import Flask, render_template

from flask_sqlalchemy import SQLAlchemy

from multiprocessing import Process



app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database/data'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
db = SQLAlchemy(app)

from database.models import DatasetCatalog, DatasetMeta


@app.route('/datacreator')
def create_project():
    # heavy_process = Process(
    #     target=testmultithread.test_function,
    #     daemon=True
    # )
    # heavy_process.start()
    # email = User.query.filter(User.id == '1').first().email

    # heavy_process = Process(
    #     target= my_func,
    #     daemon=True
    # )
    # heavy_process.start()
    # testxd = DatasetMeta.query.all()

    return render_template("index.html")


@app.route('/test')
def temp_func():

    testxd = DatasetCatalog.query.first().children
    for a in testxd:
        print(a.gender)

    return render_template("index.html")



@app.teardown_appcontext
def shutdown_session(exception=None):
    db.session.remove()


def my_func():
    pass





if __name__ == '__main__':
    app.run(threaded=True)
