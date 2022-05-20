#import Flask
import pickle
import numpy as np
from flask import Flask, render_template, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SubmitField, BooleanField
from wtforms.validators import DataRequired
import zipfile


class TraficForm(FlaskForm):
    iu_ac_5327 = BooleanField('iu_ac_5327')
    iu_ac_5328 = BooleanField('iu_ac_5328')
    iu_ac_5330 = BooleanField('iu_ac_5330')
    iu_ac_5331 = BooleanField('iu_ac_5331')
    iu_ac_5336 = BooleanField('iu_ac_5336')
    iu_ac_5370 = BooleanField('iu_ac_5370')
    iu_ac_5379 = BooleanField('iu_ac_5379')
    iu_ac_5415 = BooleanField('iu_ac_5415')
    iu_ac_5289 = BooleanField('iu_ac_5289')
    iu_ac_5324 = BooleanField('iu_ac_5324')
    periode = IntegerField('periode', validators=[DataRequired()])
    jour = IntegerField('jour', validators=[DataRequired()])
    semaine = IntegerField('semaine', validators=[DataRequired()])
    mois = IntegerField('mois', validators=[DataRequired()])
    submit = SubmitField('Predict')


#create an instance of Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'ehikvurkkfffictvefgtcltjkgfelihr'


def model(to_predict):
    with zipfile.ZipFile('randomforest_model.pkl.zip', 'r') as zip_ref:
        zip_ref.extractall()
    loaded_model = pickle.load(open("randomforest_model.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/', methods=['GET', 'POST'])
def index():
    form = TraficForm()
    if form.validate_on_submit():
        iu_ac_5327 = int(form.iu_ac_5327.data)
        iu_ac_5328 = int(form.iu_ac_5328.data)
        iu_ac_5330 = int(form.iu_ac_5330.data)
        iu_ac_5331 = int(form.iu_ac_5331.data)
        iu_ac_5336 = int(form.iu_ac_5336.data)
        iu_ac_5370 = int(form.iu_ac_5370.data)
        iu_ac_5379 = int(form.iu_ac_5379.data)
        iu_ac_5415 = int(form.iu_ac_5415.data)
        iu_ac_5289 = int(form.iu_ac_5289.data)
        iu_ac_5324 = int(form.iu_ac_5324.data)
        periode = form.periode.data
        jour = form.jour.data
        semaine = form.semaine.data
        mois = form.mois.data

        return redirect(url_for('prediction', iu_ac_5327 = iu_ac_5327,iu_ac_5328 = iu_ac_5328,iu_ac_5330 = iu_ac_5330,
                                iu_ac_5331 = iu_ac_5331,iu_ac_5336 = iu_ac_5336,iu_ac_5370 = iu_ac_5370,
                                iu_ac_5379 = iu_ac_5379, iu_ac_5415 = iu_ac_5415,iu_ac_5289 = iu_ac_5289,
                                iu_ac_5324 = iu_ac_5324, periode=periode, jour=jour,semaine=semaine, mois=mois))

    return render_template('home.html',form=form)
# , arc=arc, periode=periode, jour=jour,semaine=semaine, mois=mois
# arc, periode, jour,semaine, mois
# {{arc}} {{periode}} {{jour}} {{semaine}},{{mois}}


@app.route('/<iu_ac_5327>/<iu_ac_5328>/<iu_ac_5330>/<iu_ac_5331>/<iu_ac_5336>/<iu_ac_5370>/<iu_ac_5379>/<iu_ac_5415>'
           '/<iu_ac_5289>/<iu_ac_5324>/<periode>/<jour>/<semaine>/<mois>/prediction')
def prediction(iu_ac_5327,iu_ac_5328,iu_ac_5330,iu_ac_5331, iu_ac_5336,iu_ac_5370,
                iu_ac_5379, iu_ac_5415, iu_ac_5289,iu_ac_5324, periode, jour,semaine, mois):
    try:
        pred = model([np.array([iu_ac_5327,iu_ac_5328,iu_ac_5330,iu_ac_5331, iu_ac_5336,iu_ac_5370,
                iu_ac_5379, iu_ac_5415, iu_ac_5289,iu_ac_5324, periode, jour,semaine, mois])])

    except:
        pred= 'error with input'

    return render_template('prediction.html',iu_ac_5327 = iu_ac_5327,iu_ac_5328 = iu_ac_5328,iu_ac_5330 = iu_ac_5330,
                                iu_ac_5331 = iu_ac_5331,iu_ac_5336 = iu_ac_5336,iu_ac_5370 = iu_ac_5370,
                                iu_ac_5379 = iu_ac_5379, iu_ac_5415 = iu_ac_5415,iu_ac_5289 = iu_ac_5289,
                                iu_ac_5324 = iu_ac_5324,pred=pred, periode=periode, jour=jour,semaine=semaine, mois=mois)


if __name__ == '__main__':
    app.run(debug=True)