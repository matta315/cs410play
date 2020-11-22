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
    def read_label_data(data_dir: str, train_ff: str, test_ff: str):
        ffs = [f for f in listdir(data_dir) if isfile(join(data_dir, f)) and f.endswith('.csv')]
        lefts = []
        rights = []
        neutrals = []
        for ff in ffs:
            ff_path = join(data_dir, ff)
            df = pd.read_csv(ff_path)

            # retrieve data as a list of tuple ($Title, $Bias)
            items = df[['Title', 'Bias']].to_records(index=False)

            for item in items:
                headline = item[0]
                target = item[1].strip().lower()
                headline = Utils.clean_text_noise(headline)
                headline = Toki.tokenize_text(headline)
                line_to_write = Utils.format_for_glove(headline, target)

                if target == TAR_LEFT:
                    selected_group = lefts
                elif target == TAR_RIGHT:
                    selected_group = rights
                elif target == TAR_NEUTRAL:
                    selected_group = neutrals
                else:
                    raise Exception("Invalid target {}!".format(target))
                selected_group.append(line_to_write)

        # shuffle arrays to guarantee unbias training
        random.shuffle(lefts)
        random.shuffle(rights)
        random.shuffle(neutrals)
        print("lefts =", len(lefts))
        print("rights =", len(rights))
        print("neutrals =", len(neutrals))

        # the train/test split ratio has to applied separate to both lefts & rights, so that in either train or
        # test set, our democrat count / republican count == same
        cutoff_left = int(len(lefts) * TRAIN_RATIO)
        cutoff_right = int(len(rights) * TRAIN_RATIO)
        cutoff_neutral = int(len(neutrals) * TRAIN_RATIO)
        trains = lefts[:cutoff_left] + rights[:cutoff_right] + neutrals[:cutoff_neutral]
        tests = lefts[cutoff_left:] + rights[cutoff_right:] + neutrals[cutoff_neutral:]
        random.shuffle(trains)
        random.shuffle(tests)
        print("trains =", len(trains))
        print("tests =", len(tests))

        # now have 2 arrays trains & tests, we proceed to write them to respective files
        with open(train_ff, 'w') as ftrain:
            for ll in trains:
                ftrain.write(ll + "\n")
        with open(test_ff, 'w') as ftest:
            for ll in tests:
                ftest.write(ll + "\n")

        # build corpus
        # global thing to do: write to corpus txt for later generation of .magnitude database (word -> embedded vector)
        with open(CORPUS_TRAIN_FF, 'w') as f_corpus_train, open(CORPUS_ALL_FF, 'w') as f_corpus_all:
            for ll in trains:
                # do not include the target word
                f_corpus_train.write(ll.rsplit(' ', 1)[0] + "\n")
            t_all = lefts + rights
            random.shuffle(t_all)
            for ll in t_all:
                # do not include the target word
                f_corpus_all.write(ll.rsplit(' ', 1)[0] + "\n")
        pass

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

        # shuffle arrays to guarantee unbias training
        random.shuffle(democrats)
        random.shuffle(republics)
        print("democrats =", len(democrats))
        print("republics =", len(republics))

        # the train/test split ratio has to applied separate to both democrats & republics, so that in either train or
        # test set, our democrat count / republican count == same
        c1 = int(len(democrats) * TRAIN_RATIO)
        c2 = int(len(republics) * TRAIN_RATIO)
        trains = democrats[:c1] + republics[:c2]
        tests = democrats[c1:] + republics[c2:]
        random.shuffle(trains)
        random.shuffle(tests)
        print("trains =", len(trains))
        print("tests =", len(tests))

        # now have 2 arrays trains & tests, we proceed to write them to respective files
        with open(train_ff, 'w') as ftrain:
            for ll in trains:
                ftrain.write(ll + "\n")
        with open(test_ff, 'w') as ftest:
            for ll in tests:
                ftest.write(ll + "\n")

        # build corpus
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
