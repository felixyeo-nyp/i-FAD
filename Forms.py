import wtforms
from wtforms import Form, StringField, RadioField, SelectField, TextAreaField,FloatField, validators, ValidationError, IntegerField, FileField,DecimalField
from wtforms.fields import EmailField, DateField, PasswordField
from werkzeug.utils import secure_filename
from wtforms.validators import Email, Length, DataRequired
from wtforms import StringField, validators
from flask_wtf import FlaskForm
import re


class configurationForm(FlaskForm):

    first_timer = StringField("", validators=[DataRequired()])
    second_timer = StringField("", validators=[DataRequired()])
    pellets = IntegerField("", validators=[DataRequired(), validators.NumberRange(min=1, max=1000)])
    seconds = IntegerField("", validators=[DataRequired(), validators.NumberRange(min=60, max=1000)])
    confidence = DecimalField("", validators=[DataRequired(), validators.NumberRange(min=1, max=100)])

class RegisterForm(FlaskForm):
    username = StringField("Username", validators=[DataRequired(), Length(min=4, max=25)])
    email = EmailField("Email", validators=[DataRequired(), Email(), Length(max=150)])
    password = PasswordField("Password", validators=[DataRequired(), Length(min=8, max=80)])
    confirm_password = PasswordField("Confirm Password", validators=[
        DataRequired(), Length(min=8, max=80),
        validators.EqualTo('password', message='Passwords must match')
    ])

class LoginForm(FlaskForm):
    username = StringField("Username", validators=[DataRequired(), Length(min=4, max=25)])
    password = PasswordField("Password", validators=[DataRequired(), Length(min=8, max=80)])

class emailForm(FlaskForm):
    sender_email = StringField("", validators=[DataRequired(), Email(), Length(max=150)])
    recipient_email = StringField("", validators=[DataRequired(), Email(), Length(max=150)])
    App_password = StringField("", validators=[DataRequired(), Length(max=40)])
    days = IntegerField("", validators=[DataRequired(), validators.NumberRange(min=3, max=6)])
