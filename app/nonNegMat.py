from sklearn.decomposition import NMF
from nltk.tokenize import sent_tokenize
import numpy as np

class NonnegativeMatrixFactorization():
    def __init__(self, A, r, feature_names, corpus, no_top_words, no_top_documents):
        self.A = A
        self.r = r
        self.feature_names = feature_names
        self.corpus = corpus
        self.no_top_words = no_top_words
        self.no_top_documents = no_top_documents

    def decomposition(self):
        model = NMF(n_components=self.r, init='nndsvdar', solver='mu', beta_loss='frobenius', tol=0.1, random_state=1)
        self.W = model.fit_transform(self.A)
        self.H = model.components_
        self.fro = model.reconstruction_err_
        self.iter = model.n_iter_
        self.WH = self.W.dot(self.H)

        return self.W

    def display_summary(self):
        self.data = []
        self.data_index = []
        self.summary = []

        for self.topic_idx, self.topic in enumerate(self.H):
            self.data.append([self.feature_names[i] for i in self.topic.argsort()[:-self.no_top_words - 1:-1]])

            self.top_doc_indices = np.argsort(self.W[:, self.topic_idx])[::-1][0:self.no_top_documents]
            self.data_index.append(self.top_doc_indices)

        for self.doc_index in self.data_index[0]:
            # self.corpus[self.doc_index]
            self.summary.append(self.corpus[self.doc_index])

        self.show = '\n'.join(str(e) for e in self.summary)

        self.split = sent_tokenize(self.show)

        return self.show