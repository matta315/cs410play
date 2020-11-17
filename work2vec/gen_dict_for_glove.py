from work2vec.Utils import Utils
from work2vec.config import *


Utils.prepare_train_test_and_corpus(DATA_DIR, TRAIN_FF, TEST_FF)

"""
Now copy content of either CORPUS_TRAIN_FF or CORPUS_ALL_FF to ../glove_genvecs/text8
Then follow instruction on http://github.com/stanfordnlp/glove to generate word embedded vectors db as a .txt file
Then follow https://github.com/plasticityai/magnitude#file-format-and-converter to generate .magnitude file from .txt
"""