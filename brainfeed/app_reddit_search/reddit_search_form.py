from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SubmitField
from wtforms.fields import DateField
from wtforms.validators import DataRequired, Optional, Length, NumberRange, ValidationError
import datetime as dt

class SearchForm(FlaskForm):
    subreddit = StringField(
        'Subreddit',
        filters = [lambda x: x or None]
    )
    n_posts = IntegerField(
        'Number of Posts to be Returned',
        validators = [Optional(), NumberRange(min=0, max=20, message='Must enter an integer between 0 and 20')],
        filters = [lambda x: x or None]
    )
    start_date = DateField(
        'Start Date (search posts after this date)',
        format='%Y-%m-%d',
        validators = [Optional()],
        filters = [lambda x: x or None]
    )
    end_date = DateField(
        'End Date (search posts before this date)',
        format='%Y-%m-%d',
        validators = [Optional()],
        filters = [lambda x: x or None]
    )
    keyword = StringField(
        'Keyword',
        validators = [Optional()],
        filters = [lambda x: x or None]
    )
    
    def validate_start_date(form, field):
        if field.data > dt.date.today():
            raise ValidationError("Start date must not be later than today.")
    def validate_end_date(form, field):    
        if field.data < form.start_date.data:
            raise ValidationError("End date must not be earlier than start date.")
    
    submit = SubmitField('Search')