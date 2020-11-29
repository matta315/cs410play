import argparse

from joblib import dump, load
from pymagnitude import Magnitude
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

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


def load_transformer(word2vec, X_train=None, y_train=None):
    trans = TfidfEmbeddingVectorizer(word2vec)
    if X_train and y_train:
        print("-- loading transformer")
        trans.fit(X_train, y_train)
    return trans


def main_train():
    X_train, y_train, X_test, y_test, label_to_int, int_to_label = Utils.read_train_test_data(TRAIN_FF, TEST_FF)
    #print(X_train[0], '=', int_to_label(y_train[0]))
    #print(X_test[0], '=', int_to_label(y_test[0]))
    #exit(0)

    """
    # Here are different pre-trained glove vectors (aka dictionary) to consider
    !curl -s http://magnitude.plasticity.ai/glove+subword/glove.6B.50d.magnitude --output vectors.magnitude
    # !curl -s http://magnitude.plasticity.ai/word2vec+subword/GoogleNews-vectors-negative300.magnitude --output vectors.magnitude
    # !curl -s http://magnitude.plasticity.ai/fasttext+subword/wiki-news-300d-1M.magnitude --output vectors.magnitude
    """

    word2vec = Magnitude(CORPUS_FF)
    #word2vec = Magnitude(join(WORKING_DIR, 'glove.6B.50d.magnitude'))
    #word2vec = Magnitude(join(WORKING_DIR, 'wiki-news-300d-1M.magnitude'))

    # debug to understand what word2vec is
    #print(len(word2vec))
    #print(word2vec.dim)
    #print("dog" in word2vec)
    #print("trump" in word2vec)
    #print("democrat" in word2vec)
    #print(word2vec.query('trump'))
    #print(word2vec.distance("biden", "republican"))
    #print(word2vec.similarity("Democrats", ""))
    #print(word2vec.distance("Democrats", ["Trump", "leadership", "chairman"]))
    #print(word2vec.similarity("Democrats", ["Trump", "leadership", "chairman"]))
    #exit(0)

    #word2vec = Magnitude("glove.6B.50d.magnitude")
    #word2vec = Magnitude("wiki-news-300d-1M.magnitude")
    #print(word2vec.query("Trump"))
    #print(word2vec.dim)
    #print("dog" in word2vec)

    # create
    transformer = load_transformer(word2vec)
    svm_model = SVC(kernel='linear', C=10)
    # train pipeline
    runner = Pipeline([
        #("mean_word_vectorizer", MeanEmbeddingVectorizer(word2vec)),
        ("tf_idf_vectorizer", transformer),

        #('classifier', LogisticRegression()),
        #("extra_trees", ExtraTreesClassifier(n_estimators=200)),
        ("the_svm", svm_model)
    ])
    runner.fit(X_train, y_train)

    # save model
    dump(svm_model, open(SAVED_MODEL_FF, 'wb'))
    print('-- Model trained & saved to `{}`'.format(SAVED_MODEL_FF))


def main_test():
    X_train, y_train, X_test, y_test, label_to_int, int_to_label = Utils.read_train_test_data(TRAIN_FF, TEST_FF)
    word2vec = Magnitude(CORPUS_FF)

    # create transformer
    transformer = load_transformer(word2vec, X_train, y_train)
    transformer.fit(X_train, y_train)
    # load model
    print('-- Loading model from `{}`'.format(SAVED_MODEL_FF))
    svm_model = load(open(SAVED_MODEL_FF, 'rb'))

    # rebuild pipeline
    runner = Pipeline([
        ("tf_idf_vectorizer", transformer),
        ("the_svm", svm_model)
    ])
    predicted = runner.predict(X_test)

    # Predicting with a test dataset
    # test pipeline
    #transformer.fit(X_test, None)
    #X_test_transformed = transformer.transform(X_test)
    #predicted = svm_model.predict(X_test_transformed)
    """
    samtest = runner.predict([
        'moderate republican costello feels health care pressure in town hall'.split(' ')
        , 'here ’s the man who ’s destroying the republican party but it ’s not donald trump'.split(' ')
        , "carly fiorina was briefly a republican primary star now she 's dropping out of the race".split(' ')
    ])
    print([int_to_label(item) for item in samtest])
    exit(0)
    """
    # Model Accuracy
    print("Accuracy :", metrics.accuracy_score(y_test, predicted))
    print("Precision:", metrics.precision_score(y_test, predicted, average='weighted'))
    print("Recall   :", metrics.recall_score(y_test, predicted, average='weighted'))
    print("F1 score :", metrics.f1_score(y_test, predicted, average='weighted'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='check to generate DAGs...')
    parser.add_argument("--train", action='store_true', help="add this flag run TRAIN")
    parser.add_argument("--test", action='store_true', help="add this flag run TEST")
    args = parser.parse_args()
    if args.train:
        print('-- training..')
        main_train()
    if args.test:
        print('-- testing..')
        main_test()
