from app import app
from flask import render_template, request, redirect, url_for
from flask_mysqldb import MySQL
import yaml
from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import os
from werkzeug.utils import secure_filename
import pandas as pd
from app import prepro
from app import nonNegMat

# Configure db
db = yaml.load(open('app/yaml/db.yaml'))
app.config['MYSQL_HOST'] = db['mysql_host']
app.config['MYSQL_USER'] = db['mysql_user']
app.config['MYSQL_PASSWORD'] = db['mysql_password']
app.config['MYSQL_DB'] = db['mysql_db']

mysql = MySQL(app)

# Pdf Miner
def convert(fname, pages=None):
	if not pages:
		pagenums = set()
	else:
		pagenums = set(pages)

	output = StringIO()
	manager = PDFResourceManager()
	converter = TextConverter(manager, output, laparams=LAParams())
	interpreter = PDFPageInterpreter(manager, converter)

	for page in PDFPage.get_pages(fname, pagenums):
		interpreter.process_page(page)

	converter.close()
	text = output.getvalue()
	output.close

	return text

@app.route('/')

@app.route('/index')
def index():

    return render_template('index.html', title='Peringkas Teks Otomatis')

@app.route('/proses', methods=['POST', 'GET'])
def proses():
	if request.method == 'POST':
		journal = request.form['title']
		file = request.files['file']
		option = request.form['optradio']
		read_pdf = convert(file)
		corpus = read_pdf.replace('\n', ' ')

		# text preprocessing
		preprocessing = prepro.Preprocessing(corpus)
		sentences = preprocessing.sent_preprocessing()
		m = preprocessing.m
		n = preprocessing.n
		feature_names = preprocessing.feature_names
		sent_ind = preprocessing.sent_ind

		# menentukan nilai r
		if m < n:
			r = m
		elif m > n:
			r = n
		elif m == n:
			r = m

		A = preprocessing.A
		no_top_words = 5
		no_top_documents = int(option)

		nmf = nonNegMat.NonnegativeMatrixFactorization(A, r, feature_names, sent_ind, no_top_words, no_top_documents)
		decomposition = nmf.decomposition()
		frobenius_norm = nmf.fro
		iteration = nmf.iter

		display_summary = nmf.display_summary()
		top_word = nmf.data[0]
		summary = nmf.split

	return render_template('proses.html', title='Proses & Hasil Ringkasan - Peringkas Teks Otomatis', journal=journal, corpus=corpus, 
		sentences=sentences, frobenius=frobenius_norm, iteration=iteration, top_word=top_word, summary=summary, n=n, m=m)

@app.route('/stopword')
def stopword():
	cur = mysql.connection.cursor()
	resultValue = cur.execute("SELECT * FROM stopword LIMIT 100")
	if resultValue > 0:
		stopWords = cur.fetchall()

		return render_template('stopword.html', title='Stopword - Peringkas Teks Otomatis', stopWords=stopWords)

@app.route('/katadasar')
def katadasar():
	cur = mysql.connection.cursor()
	resultValue = cur.execute("SELECT * FROM katadasar LIMIT 100")
	if resultValue > 0:
		kataDasar = cur.fetchall()

		return render_template('katadasar.html', title='Kata Dasar - Peringkas Teks Otomatis', kataDasar=kataDasar)

@app.route('/tentang')
def tentang():
	
	return render_template('tentang.html', title='Tentang - Peringkas Teks Otomatis')

@app.route("/tables_preprocessing")
def tables_preprocessing():
	table_token = pd.read_excel('app/data/hasil tokenizing.xlsx')
	table_token.index.name=None

	return render_template('preprocessing.html', tables_token=[table_token.to_html(classes='co')])

@app.route('/tables_weighting')
def table_weighting():
	table_weighting = pd.read_excel('app/data/hasil frekuensi kata.xlsx')
	table_weighting.index.name=None

	return render_template('pembobotan.html', tables_weighting=[table_weighting.to_html(classes='co')])

@app.route('/tables_nmf')
def table_nmf():
	table_nmf = pd.read_excel('app/data/WH.xlsx')
	table_nmf.index.name=None

	return render_template('nmf.html', tables_nmf=[table_nmf.to_html(classes='co')])