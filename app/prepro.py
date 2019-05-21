import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from itertools import chain
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

class Preprocessing():
	"""docstring for Preprocessing"""

	def __init__(self, corpus):
		self.corpus = corpus

	def sent_preprocessing(self):
		
		# pemecahan dokumen menjadi per kalimat
		sentences = sent_tokenize(self.corpus.lower())

		# menghilangkan karakter spesial
		data = []
		for i in sentences:
			data.append(re.sub(r'-(?:(?<!\b[0-9]{4}-)|(?![0-9]{2}(?:[0-9]{2})?\b))', ' ', i))
		removetable = str.maketrans('', '', '@#%()[]{}.,!?><:;*&^+=_`~$"|/')
		clean_sent = [s.translate(removetable) for s in data]

		# filtering (menghilangkan stopwords)
		stopwords_id = set(stopwords.words('indonesian'))
		sent_stopwords = []
		for sent in clean_sent:
			sent_stopwords.append(' '.join(w for w in nltk.word_tokenize(sent) if w.lower() not in stopwords_id))

		sw = pd.DataFrame(sorted(sent_stopwords))
		sw.to_excel('app/data/hasil filtering.xlsx', index=False, header=False)

		# stemming bahasa indonesia
		factory = StemmerFactory()
		stemmer = factory.create_stemmer()

		stemming = []
		for sent in sent_stopwords:
			stemming.append(stemmer.stem(sent))

		stm = pd.DataFrame(stemming, columns=['Stemming'])
		stm.to_excel('app/data/hasil stemming.xlsx', index=False, header=True)

		freq_vectorizer = CountVectorizer()
		freq = freq_vectorizer.fit_transform(stemming)
		freq_feature_names = freq_vectorizer.get_feature_names()

		ffn = pd.DataFrame(sorted(freq_feature_names), columns=['Tokenizing, Filtering & Stemming'])
		ffn.to_excel('app/data/hasil tokenizing.xlsx', index=False, header=True)

		freq_mat = pd.DataFrame(freq.todense(), columns=freq_feature_names)
		freq_mat.to_excel('app/data/hasil frekuensi kata.xlsx', index=False)

		self.m = len(freq_mat.columns)
		self.n = len(freq_mat.index)
		self.A = freq
		self.feature_names = freq_feature_names
		self.sent_ind = sentences

		return freq_mat