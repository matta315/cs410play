from os import listdir
from os.path import isfile, join
import random

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
    return "BOS {} EOS {}\n".format(text, target)


def prepare_clean_corpus(data_dir: str, output_txt: str):
    ffs = [f for f in listdir(data_dir) if isfile(join(data_dir, f)) and f.endswith('.csv')]
    with open(output_txt, 'w') as fw:
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
            for title in titles:
                strline = clean_text_noise(title)
                strline = format_for_glove(strline, target)
                #print(to_write)
                fw.write(strline)
    pass


def train_test_split(corpus_ff: str, train_ff: str, train_ratio: float, test_ff: str):
    with open(corpus_ff, mode='rb') as f:
        lines = f.readlines()
    lines = [l.decode('utf-8').strip() for l in lines]
    #print(lines[-1])
    #print(len(lines))
    random.shuffle(lines)
    cutoff = int(len(lines) * train_ratio)
    with open(train_ff, 'w') as ftrain:
        for ll in lines[:cutoff]:
            ftrain.write(ll+"\n")
    with open(test_ff, 'w') as ftest:
        for ll in lines[cutoff:]:
            ftest.write(ll+"\n")
    pass


def main():
    data_dir = './rawdata'
    corpus_ff = './corpus.txt'
    train_ff = './train.txt'
    test_ff = './test.txt'
    #prepare_clean_corpus(data_dir, corpus_ff)

    #train_test_split(corpus_ff, train_ff, TRAIN_RATIO, test_ff)
    pass


if __name__ == '__main__':
    main()
