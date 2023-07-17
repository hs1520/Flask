from exts import db
from datetime import datetime

# ORM mode
class ClassificationModel(db.Model):
    __tablename__ = "module"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    file_name = db.Column(db.String(30), primary_key=True)  # record module code
    prediction = db.Column(db.String(10), nullable=False)  # record module title
    score = db.Column(db.Float, nullable=False)
    index = db.Column(db.Integer, nullable=False)
    time = db.Column(db.DateTime,default=datetime.now)
