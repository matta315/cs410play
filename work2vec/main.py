from collections import defaultdict

import numpy as np
from pymagnitude import Magnitude
from sklearn.feature_extraction.text import TfidfVectorizer

from work2vec.Utils import Utils
from work2vec.config import *

"""
Trying this method
http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
- Train our own model from the corpus of political titles
- Democrat   = 1
- Republican = 0
In combination with this work
https://colab.research.google.com/drive/1lOcAhIffLW8XC6QsKzt5T_ZqPP4Y9eS4#scrollTo=KGPuY8DByPyU
"""


# Credit http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
class MeanEmbeddingVectorizer(object):
    word2vec = None
    dim = None

    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = word2vec.dim

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            #np.mean([self.word2vec.query(w) for w in words if w in self.word2vec]
            #        or [np.zeros(self.dim)], axis=0)
            np.mean([self.word2vec.query(w) for w in words], axis=0)
            for words in X
        ])


# Credit http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = word2vec.dim

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()]
        )

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec.query(w) * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])


def main():
    do_prepare_train_test = False
    if do_prepare_train_test:
        Utils.prepare_train_test(DATA_DIR, TRAIN_FF, TEST_FF)
        exit(0)

    # Now: proceed to create the .magnitude file manually, then come back here and resume

    X_train, y_train, X_test, y_test, party_to_int, int_to_party = Utils.read_train_test_data(TRAIN_FF, TEST_FF)
    #print(X_train[0], '=', int_to_party(y_train[0]))
    #print(X_test[0], '=', int_to_party(y_test[0]))

    word2vec = Magnitude("./vectors.magnitude")
    #word2vec = Magnitude("glove.6B.50d.magnitude")
    #print(word2vec.query("Trump"))
    #print(word2vec.dim)
    #print("dog" in word2vec)

    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn import metrics

    runner = Pipeline([
        ("mean_word_vectorizer", MeanEmbeddingVectorizer(word2vec)),
        #("tf_idf_vectorizer", TfidfEmbeddingVectorizer(word2vec)),
        ("extra_trees", ExtraTreesClassifier(n_estimators=200))
    ])
    runner.fit(X_train, y_train)

    # Predicting with a test dataset
    predicted = runner.predict(X_test)

    """
    samtest = runner.predict([
        'BOS Rejecting Trump, Wall Street Republican donors scatter largesse EOS'.split(' ')
        , 'BOS With one bill, Republicans fast track plan to undo Obama regulations EOS'.split(' ')
        , 'BOS Democrats fret unions’ pressure tactics on trade EOS'.split(' ')
    ])
    print([int_to_party(item) for item in samtest])
    """

    # Model Accuracy
    print("Accuracy :", metrics.accuracy_score(y_test, predicted))
    print("Precision:", metrics.precision_score(y_test, predicted))
    print("Recall   :", metrics.recall_score(y_test, predicted))
    print("F1 score :", metrics.f1_score(y_test, predicted))

    pass


if __name__ == '__main__':
    main()
