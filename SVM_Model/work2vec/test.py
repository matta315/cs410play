import pandas as pd
from os import listdir
from os.path import isfile, join #Use to join path + Dir
from SVM_Model.work2vec.config import *
from SVM_Model.work2vec.Tokenizer import Tokenizer as Toki

def read_data (dir: str):
    files = listdir(dir)
    with open (CORPUS_ALL_FF,'w+') as fcorpus, open (ALL_FF,'w+') as fAll :
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
            else:
                for tt in titles:
                    try:
                        title = tt[0] + ',' + labels
                        fAll.write(title + "\n")
                    except:
                        continue

def split_train_test (train_ff, test_ff, file_all):
    lable_names = dict()
    with open(file_all, 'rb') as ff:
        lines = [line.decode('utf-8') for line in ff.readlines()]
    #print(lines[0])
    for line in lines:
        title = line.rsplit(',',1)[0]
        lable = line.rsplit(',',1)[1]
        if lable not in lable_names:
            lable_names[lable] = []
        lable_names[lable].append(title)

#????????? CHECK IN!
    cutoffs = [int(len(lable_names[name]) * TRAIN_RATIO) for name in group_names]
    trains = []
    tests = []
    for name, cof in zip(group_names, cutoffs):
        gg = target_to_data[name]
        trains += gg[:cof]
        tests += gg[cof:]
    random.shuffle(trains)
    random.shuffle(tests)
    print("trains =", len(trains))
    print("tests =", len(tests))

#read_data('../../data_raw')
split_train_test(TRAIN_FF,TEST_FF,ALL_FF)
