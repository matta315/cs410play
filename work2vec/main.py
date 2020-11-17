from os import listdir
from os.path import isfile, join
import random

from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier

from pymagnitude import MagnitudeUtils

import pandas as pd

"""
Trying this method
http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
- Train our own model from the corpus of political titles
- Democrat   = 1
- Republican = 0
In combination with this work
https://colab.research.google.com/drive/1lOcAhIffLW8XC6QsKzt5T_ZqPP4Y9eS4#scrollTo=KGPuY8DByPyU
"""

TAR_DEMOCRAT = 'democrat'
TAR_REPUBLIC = 'republic'

TRAIN_RATIO = 0.6


# TODO
def clean_text_noise(tt: str):
    return tt.strip()


# Exmaple train file: http://magnitude.plasticity.ai/data/atis/atis-intent-train.txt
# Exmaple test file : http://magnitude.plasticity.ai/data/atis/atis-intent-test.txt
def format_for_glove(text: str, target: str):
    return "BOS {} EOS {}".format(text, target)


def prepare_train_test(data_dir: str, train_ff: str, test_ff: str):
    ffs = [f for f in listdir(data_dir) if isfile(join(data_dir, f)) and f.endswith('.csv')]
    democrats = []
    republics = []
    for ff in ffs:
        ff_path = join(data_dir, ff)
        #print(ff_path)
        if ff_path.endswith('_D.csv'):
            target = TAR_DEMOCRAT
        elif ff_path.endswith('_R.csv'):
            target = TAR_REPUBLIC
        else:
            raise Exception("Error reading {}!".format(ff_path))
        df = pd.read_csv(join(data_dir, ff))
        titles = df['title'].to_list()
        selected_group = democrats if target == TAR_DEMOCRAT else republics
        for title in titles:
            strline = clean_text_noise(title)
            strline = format_for_glove(strline, target)
            selected_group.append(strline)

    random.shuffle(democrats)
    random.shuffle(republics)
    print("democrats =", len(democrats))
    print("republics =", len(republics))
    c1 = int(len(democrats) * TRAIN_RATIO)
    c2 = int(len(republics) * TRAIN_RATIO)
    trains = democrats[:c1] + republics[:c2]
    tests = democrats[c1:] + republics[c2:]
    random.shuffle(trains)
    random.shuffle(tests)
    print("trains =", len(trains))
    print("tests =", len(tests))
    with open(train_ff, 'w') as ftrain:
        for ll in trains:
            ftrain.write(ll+"\n")
    with open(test_ff, 'w') as ftest:
        for ll in tests:
            ftest.write(ll+"\n")
    pass


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
    data_dir = './rawdata'
    train_ff = './train.txt'
    test_ff = './test.txt'
    #prepare_train_test(data_dir, train_ff, test_ff)

    X_train, y_train, X_test, y_test, party_to_int, int_to_party = read_train_test_data(train_ff, test_ff)
    print(X_train[0])
    print(int_to_party(y_train[0]))
    print(X_test[0])
    print(int_to_party(y_test[0]))

    pass


if __name__ == '__main__':
    main()
