import pandas as pd
from os import listdir
from os.path import isfile, join #Use to join path + Dir
from SVM_Model.work2vec.config import *
from SVM_Model.work2vec.Tokenizer import Tokenizer as Toki

def read_data (dir: str):
    files = listdir(dir)
    with open (CORPUS_ALL_FF,'w+') as fcorpus:
        for ff in files:
            dd = join(dir, ff)
            labels = dd.replace('.csv', '').rsplit('_', 1)[1].lower()
            titles = pd.read_csv(dd)[["title"]].to_records(index=False) #return a tuble
            print(dd)
            if labels == 'all':
                for tt in titles:
                    try:
                        title = tt[0]
                        fcorpus.write(Toki.normalize_text(title) + "\n")
                    except:
                        continue
        #Build corpus





read_data('../../data_raw')
