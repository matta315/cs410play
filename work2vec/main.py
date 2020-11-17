from pymagnitude import MagnitudeUtils

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


def read_train_test_data(train_ff: str, test_ff: str):
    with open(train_ff, 'rb') as ff:
        trains = [line.decode('utf-8') for line in ff.readlines()]
    with open(test_ff, 'rb') as ff:
        tests = [line.decode('utf-8') for line in ff.readlines()]

    """
        etree_w2v = Pipeline([
            ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
            ("extra trees", ExtraTreesClassifier(n_estimators=200))])
        """
    add_party, party_to_int, int_to_party = MagnitudeUtils.class_encoding()

    X_train = [line.split(" ")[0:-1] for line in trains]
    y_train = [add_party(line.split(" ")[-1]) for line in trains]
    X_test = [line.split(" ")[0:-1] for line in tests]
    y_test = [add_party(line.split(" ")[-1]) for line in tests]

    return X_train, y_train, X_test, y_test, party_to_int, int_to_party


def main():
    Utils.prepare_train_test(DATA_DIR, TRAIN_FF, TEST_FF)

    X_train, y_train, X_test, y_test, party_to_int, int_to_party = read_train_test_data(TRAIN_FF, TEST_FF)
    print(X_train[0], '=', int_to_party(y_train[0]))
    print(X_test[0], '=', int_to_party(y_test[0]))

    pass


if __name__ == '__main__':
    main()
