from os import listdir
from os.path import isfile, join

import pandas as pd

"""
Trying this method
http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
- Train our own model from the corpus of political titles
- Democrat   = 1
- Republican = 0
"""

TAR_DEMOCRAT = 1
TAR_REPUBLIC = 0


# TODO
def clean_text_noise(tt: str):
    return tt.strip()


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
                to_write = clean_text_noise(title)
                #print(to_write)
                fw.write("{} {}\n".format(target, to_write))
    pass


def main():
    data_dir = './rawdata'
    output_ff = './corpus.txt'
    prepare_clean_corpus(data_dir, output_ff)
    print("Corpus write to:", output_ff)
    pass


if __name__ == '__main__':
    main()
