

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
sample = "Stephen is Awesome and Great ):"
print (sample)
print (vader.polarity_scores(sample))

#Initialize Pandas Dataframe with test data
df = pd.read_csv("/Users/nbachman/Documents/HCP Anywhere/GradSchool/Text Mining and Analytics/CS410-BiasDetector/data_raw/NYT_D.csv")

#Add a sentiment score of the title column to dataframe and print top 5
df["scores"] = df["title"].apply(lambda title: vader.polarity_scores(title))


#Add a column with just the compound value
df["compound"] = df["scores"].apply(lambda score_dict: score_dict["compound"])
#print (df)

#Add a sentiment column for pos, neg or neutral
df["sentiment"] = df["compound"].apply(lambda c: "positive" if c > 0 else ("negative" if c < 0 else "neutral"))
#print (df.head())

#Print the number of neutral, positve and negative headlines
#print (df["sentiment"].value_counts())