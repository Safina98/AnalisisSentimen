
import os
from flask import Flask, render_template, send_file, request, abort
from werkzeug.utils import secure_filename

from ipynb.fs.full.ProgramClassess import DataPreparation, SentimentScore,TFIDFValue,CART,RandomForest,NaiveBayes

import numpy as np
import pandas as pd
import pickle

from googletrans import Translator
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from collections import Counter
from matplotlib.figure import Figure
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
pd.set_option('display.max_columns', None)

app = Flask(__name__)
app.config['UPLOAD_EXTENSIONS'] = ['.csv']
app.config['UPLOAD_PATH'] = 'uploads'

def detect_sarcasm(X_test):
	p = DataPreparation()
	X_test = p.extractEmoji(X_test) # Convert emoji to word
	X_test = p.splitHashtag(X_test) # Split hashtag
	processed_data_test = p.preprocessData(X_test)# Preprocess data
	translated_datatest = p.translateTweet(processed_data_test) #Translate to english
	first_halftest, last_halftest = p.splitTweet(translated_datatest) # Split to firt half and last half
	s = SentimentScore() 
	firsthalftest_score = s.main(first_halftest) #first half tweet sentiment score 
	lasthalftest_score = s.main(last_halftest) #first half tweet sentiment score 
	interjeksitest = p.extractInterjection(X_test) #number of interjection words in tweet
	featuretest = np.transpose(np.array([firsthalftest_score,lasthalftest_score,interjeksitest]))
	featuretest = pd.DataFrame(featuretest, columns =['firsthalf_score','lasthalf_score','interjeksi'])
	f = open('randomforest.pickle', 'rb')
	fr = pickle.load(f)
	f.close()
	rf = RandomForest()
	predrf = rf.random_forest_predictions(fr,np.array(featuretest))
	return predrf

def sentiment_analysis(X_test,predrf):
	p = DataPreparation()
	processed_data_test = p.preprocessData(X_test)# Preprocess data
	datatest_without_stopword = p.removeStopword(processed_data_test) #Remove stopword
	stemmed_datatest = p.stemmingTweet(datatest_without_stopword) #stemming
	#TF-IDF
	f = open('words3.pickle', 'rb')
	words = pickle.load(f)
	f.close()
	v = TFIDFValue()
	tf_idf_test = v.main(words,stemmed_datatest)
	f = open('naivebayes3.pickle', 'rb')
	vl = pickle.load(f)
	f.close()
	nb = NaiveBayes()
	pred = nb.naive_bayes(vl,tf_idf_test)
	pred_imp = pred
	for i in range(len(pred)):
		if (predrf[i]=='sarcasm'):
			pred_imp[i] = 'Negative'
	return pred_imp
def create_fig(pred):
	fig = Figure()
	fig.patch.set_alpha(0.)
	axis = fig.add_subplot(1, 1, 1)
	axis.set_yticks(np.array(list(Counter(pred).values())))
	axis.bar(list(Counter(pred).keys()), list(Counter(pred).values()),color=['grey'])
	pngImage = io.BytesIO()
	FigureCanvas(fig).print_png(pngImage)
	pngImageB64String = "data:image/png;base64,"
	pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')
	return pngImageB64String
def sent_fig(pred_):
	fig = Figure()
	fig.patch.set_alpha(0.)
	axis = fig.add_subplot(1, 1, 1)
	axis.set_yticks(np.array(list(Counter(pred_).values())))
	y_ax = []
	x_ax = ['Positive','Neutral','Negative']
	for x in x_ax:
		 y_ax.append(pred_.count(x))
	axis.bar( ['Positive','Neutral','Negative'], y_ax,color=['grey'])
	pngImage = io.BytesIO()
	FigureCanvas(fig).print_png(pngImage)
	pngImageB64String = "data:image/png;base64,"
	pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')
	return pngImageB64String


@app.route('/')
def home():
	files = os.listdir(app.config['UPLOAD_PATH'])
	return render_template('home.html', files=files)

@app.route('/', methods=['POST'])
def upload_files():
	uploaded_file = request.files['file']
	filename = secure_filename(uploaded_file.filename)
	if filename != '':
		file_ext = os.path.splitext(filename)[1]
		if file_ext not in app.config['UPLOAD_EXTENSIONS']:
			abort(400)
		uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
		data_test = pd.read_csv(os.path.join(app.config['UPLOAD_PATH'], filename))
		X_test = data_test['Tweet'].values
		predrf = detect_sarcasm(X_test)
		data_test['Sarcasm'] = predrf
		pred = sentiment_analysis(X_test,predrf)
		data_test['Sentiment'] = pred
		data_test.to_csv(os.path.join(app.config['UPLOAD_PATH'], 'sentiment.csv'),index = False)
		pred_img = sent_fig(pred)
		predrf_img = create_fig(predrf)
	return render_template('tabletemp.html',  tables=[data_test.to_html(classes='data')],predimage=pred_img,predrfimg=predrf_img)
@app.route('/return-files/')
def return_files_tut():
	try:
		path = os.path.join(app.config['UPLOAD_PATH'], 'sentiment.csv')
		return send_file(path, as_attachment=True)
	except Exception as e:
		return str(e)
@app.route('/return-template/')
def return_template_tut():
	try:
		path = os.path.join(app.config['UPLOAD_PATH'], 'Template.csv')
		return send_file(path, as_attachment=True)
	except Exception as e:
		return str(e)
