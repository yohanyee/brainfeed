from flask import Flask
# from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SECRET_KEY'] = 'a1d992e58341524e3799a39c9da41a96'

# Database
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# db = SQLAlchemy(app)

# Session
# app.config['SESSION_TYPE'] = 'sqlalchemy'
# app.config['SESSION_SQLALCHEMY_TABLE'] = 'sessions'
# app.config['SESSION_SQLALCHEMY'] = db
# Session(app) # session = Session(app)
# session.app.session_interface.db.create_all()

app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"

from app_reddit_search import routes
