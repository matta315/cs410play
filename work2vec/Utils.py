import random
from os import listdir
from os.path import isfile, join

from work2vec.config import *

import pandas as pd


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
        return "BOS {} EOS {}".format(text, target)

    @staticmethod
    def prepare_train_test(data_dir: str, train_ff: str, test_ff: str):
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
                strline = Utils.clean_text_noise(title)
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
        pass

