from flask import Flask, render_template
from database.database import db_session
from database.models import DatasetCatalog, DatasetMeta
import test
from multiprocessing import Process

app = Flask(__name__)


@app.route('/datacreator')
def create_project():
    # heavy_process = Process(
    #     target=test.testxd,
    #     daemon=True
    # )
    # heavy_process.start()
    # email = User.query.filter(User.id == '1').first().email

    print(db_session.query(DatasetCatalog))
    return render_template("index.html")


@app.teardown_appcontext
def shutdown_session(exception=None):
    db_session.remove()


def my_func():
    print("Process finished")


if __name__ == '__main__':
    app.run(threaded=True)
