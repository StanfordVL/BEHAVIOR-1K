from flask import Flask
from . import filters

def create_app():
    app = Flask(__name__)
    app.register_blueprint(filters.bp)
    return app
