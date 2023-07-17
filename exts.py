# to figure the deadlock problem
from sqlalchemy import create_engine
from config import DB_URI
from sqlalchemy.orm import sessionmaker
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

# create ORM model and map them into the database
engine = create_engine(DB_URI)  # create engine
session = sessionmaker(engine)()  # create session


