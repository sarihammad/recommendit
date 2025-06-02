import os

from flask import Flask

from app.routes import auth

from . import db
from app.routes.book import book_routes


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'recommendit.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    
    # @app.route('/hello')
    # def hello():
    #     return 'Hello, World!'
    
    db.init_app(app)
    app.register_blueprint(book_routes)
    app.register_blueprint(auth)

    return app