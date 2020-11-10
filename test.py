

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

#Create an instance of SentimentIntesityAnalyzer called vader
vader = SentimentIntensityAnalyzer() #

#Test a sample of text
#sample = "All Hell Breaks Loose at the Democratic Convention"
#print (vader.polarity_scores(sample))

#Initialize Pandas Dataframe with test data
df = pd.read_csv("/Users/nbachman/Documents/HCP Anywhere/GradSchool/Text Mining and Analytics/CS410-BiasDetector/data/ATL_D.csv")

#Add a sentiment score of the title column to dataframe and print top 5
df["scores"] = df["title"].apply(lambda title: vader.polarity_scores(title))
print (df.head())