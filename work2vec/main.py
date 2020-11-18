from pymagnitude import Magnitude
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier

from work2vec.Utils import Utils
from work2vec.config import *

from work2vec.text_vector import *

"""
Trying this method
http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
- Train our own model from the corpus of political titles
- Democrat   = 1
- Republican = 0
In combination with this work
https://colab.research.google.com/drive/1lOcAhIffLW8XC6QsKzt5T_ZqPP4Y9eS4#scrollTo=KGPuY8DByPyU

test some change
asdfdsf a

asdf
asd
f
"""


def main():
    do_prepare_train_test = False
    if do_prepare_train_test:
        Utils.prepare_train_test_and_corpus(DATA_DIR, TRAIN_FF, TEST_FF)
        exit(0)

    # Now: proceed to create the .magnitude file manually, then come back here and resume

    X_train, y_train, X_test, y_test, party_to_int, int_to_party = Utils.read_train_test_data(TRAIN_FF, TEST_FF)
    #print(X_train[0], '=', int_to_party(y_train[0]))
    #print(X_test[0], '=', int_to_party(y_test[0]))

    """
    # Here are different pre-trained glove vectors (aka dictionary) to consider
    !curl -s http://magnitude.plasticity.ai/glove+subword/glove.6B.50d.magnitude --output vectors.magnitude
    # !curl -s http://magnitude.plasticity.ai/word2vec+subword/GoogleNews-vectors-negative300.magnitude --output vectors.magnitude
    # !curl -s http://magnitude.plasticity.ai/fasttext+subword/wiki-news-300d-1M.magnitude --output vectors.magnitude
    """

    word2vec = Magnitude("./vectors.magnitude")
    #word2vec = Magnitude("glove.6B.50d.magnitude")
    #word2vec = Magnitude("wiki-news-300d-1M.magnitude")
    #print(word2vec.query("Trump"))
    #print(word2vec.dim)
    #print("dog" in word2vec)

    from sklearn.pipeline import Pipeline
    from sklearn import metrics

    runner = Pipeline([
        ("mean_word_vectorizer", MeanEmbeddingVectorizer(word2vec)),
        #("tf_idf_vectorizer", TfidfEmbeddingVectorizer(word2vec)),
        #('classifier', LogisticRegression()),
        ("extra_trees", ExtraTreesClassifier(n_estimators=200)),
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
