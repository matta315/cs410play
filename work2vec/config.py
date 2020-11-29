import os
from os.path import join

WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
#print(join(THIS_DIR, '../rawdata'))

TAR_DEMOCRAT = 'democrat'
TAR_REPUBLIC = 'republic'

TRAIN_RATIO = 0.7

DATA_DIR = join(WORKING_DIR, '../rawdata')
LABELED_DATA_DIR = join(WORKING_DIR, '../labeled_data/politics')
#LABELED_DATA_DIR = '../labeled_data/airline'
TRAIN_FF = join(WORKING_DIR, './train.txt')
TEST_FF = join(WORKING_DIR, './test.txt')

CORPUS_TRAIN_FF = join(WORKING_DIR, 'corpus-train.txt')
CORPUS_ALL_FF = join(WORKING_DIR, 'corpus-all.txt')

CORPUS_FF = join(WORKING_DIR, './vectors.magnitude')

SAVED_MODEL_FF = join(WORKING_DIR, './model/model-pipeline.joblib')


class Config(object):
    def __init__(self):
        pass
