

import nltk
import ssl
import numpy as np
import pandas as pd

#only need to run first time
#try:
#    _create_unverified_https_context = ssl._create_unverified_context
#except AttributeError:
#    pass
#else:
#    ssl._create_default_https_context = _create_unverified_https_context

#nltk.download("vader_lexicon") # downloads vader_lexicon to /Users/nbachman/nltk_data

from nltk.sentiment.vader import SentimentIntensityAnalyzer
#print("import nltk,vader and SentimentIntesity complete")

vader = SentimentIntensityAnalyzer() #Create an instance of SentimentIntesityAnalyzer called vader

sample = "All Hell Breaks Loose at the Democratic Convention"
print (vader.polarity_scores(sample))