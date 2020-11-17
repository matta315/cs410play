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


def main():
    #Utils.prepare_train_test(DATA_DIR, TRAIN_FF, TEST_FF)

    X_train, y_train, X_test, y_test, party_to_int, int_to_party = Utils.read_train_test_data(TRAIN_FF, TEST_FF)
    print(X_train[0], '=', int_to_party(y_train[0]))
    print(X_test[0], '=', int_to_party(y_test[0]))

    pass


if __name__ == '__main__':
    main()
