from app_reddit_search import db

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True)
    
    def __repr__(self):
        return f"User('{self.username}')"