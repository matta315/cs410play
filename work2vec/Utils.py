import random
from os import listdir
from os.path import isfile, join

import pandas as pd
from pymagnitude import MagnitudeUtils

from work2vec.Tokenizer import Tokenizer as Toki
from work2vec.config import *


class Utils(object):
    def __init__(self):
        pass

    # TODO
    @staticmethod
    def clean_text_noise(tt: str):
        return tt.strip()

    # Exmaple train file: http://magnitude.plasticity.ai/data/atis/atis-intent-train.txt
    # Exmaple test file : http://magnitude.plasticity.ai/data/atis/atis-intent-test.txt
    @staticmethod
    def format_for_glove(text: str, target: str):
        #return "BOS {} EOS {}".format(text, target)
        return "{} {}".format(text, target)

    @staticmethod
    def prepare_train_test_and_corpus(data_dir: str, train_ff: str, test_ff: str):
        ffs = [f for f in listdir(data_dir) if isfile(join(data_dir, f)) and f.endswith('.csv')]
        democrats = []
        republics = []
        for ff in ffs:
            ff_path = join(data_dir, ff)
            # print(ff_path)
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
                strline = title
                strline = Utils.clean_text_noise(strline)
                strline = Toki.tokenize_text(strline)
                strline = Utils.format_for_glove(strline, target)
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
                ftrain.write(ll + "\n")
        with open(test_ff, 'w') as ftest:
            for ll in tests:
                ftest.write(ll + "\n")

        # global thing to do: write to corpus txt for later generation of .magnitude database (word -> embedded vector)
        with open(CORPUS_TRAIN_FF, 'w') as f_corpus_train, open(CORPUS_ALL_FF, 'w') as f_corpus_all:
            for ll in trains:
                # do not include the target word
                f_corpus_train.write(ll.rsplit(' ', 1)[0] + "\n")
            t_all = democrats + republics
            random.shuffle(t_all)
            for ll in t_all:
                # do not include the target word
                f_corpus_all.write(ll.rsplit(' ', 1)[0] + "\n")
        pass

    @staticmethod
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
        add_party, label_to_int, int_to_label = MagnitudeUtils.class_encoding()

        X_train = [line.split(" ")[0:-1] for line in trains]
        y_train = [add_party(line.split(" ")[-1]) for line in trains]
        X_test = [line.split(" ")[0:-1] for line in tests]
        y_test = [add_party(line.split(" ")[-1]) for line in tests]

        return X_train, y_train, X_test, y_test, label_to_int, int_to_label

