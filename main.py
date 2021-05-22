#!/usr/bin/python
# -*- coding: utf-8 -*-
from flask import Flask, render_template, flash, request, Markup, send_file
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import pandas as pd
import sys
import json
import os
import shutil
from os import path
from os import urandom
import urllib
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


path = path.dirname(path.realpath(__file__))
app = Flask(__name__,template_folder=path+'')


#app.config['SECRET_KEY'] = '3d441f27u331c27333d331k2f3333a'
app.config['SECRET_KEY'] = urandom(24)

df = pd.read_csv('moviedatabase.csv', sep=',' , error_bad_lines=False, index_col=False, dtype='object', usecols=['title','genres', 'keywords', 'popularity', 'cast', 'director'])
df.head()

# df.columns
features = ['genres', 'keywords', 'popularity', 'cast', 'director']
data = pd.DataFrame(df['title'])
for f in features:
    temp = pd.DataFrame(df[f])
    data = pd.concat([data, temp], axis=1)

del df

data.fillna('', inplace=True)


dataX = data.drop(['title', 'popularity'], axis=1)
dataX["combined_features"] = dataX['genres'] + " " + dataX['keywords'] + " " + dataX['cast'] + " " + dataX['director']

cv = CountVectorizer()
cv_mat = cv.fit_transform(dataX['combined_features']).toarray()

similarity = cosine_similarity(cv_mat)


class ReusableForm(Form):

    movie = TextField("Movie name?")
    #github = TextField("What's your Github handle?")



@app.route("/", methods=['GET', 'POST'])
def hello():
    form= ReusableForm(request.form)
    if request.method == 'POST':
        mname=request.form['mname']
        i = data[data['title'] == mname].index
        i = i[0]
        indexed_sim = list(enumerate(similarity[i]))
        sorted_sim = sorted(indexed_sim, key=lambda x: x[1], reverse=True)
        req_sim = sorted_sim[1:15]
        movies=[]
        for i, _ in req_sim:
            movies.append(data['title'][i])
            flash(data['title'][i])
        return render_template('index.html',mname=movies)

    elif request.method == 'GET':
        return render_template('index.html')



if __name__ == "__main__":
	app.run(host='localhost')

