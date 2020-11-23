import nltk
import ssl
import numpy as np
import pandas as pd
import difflib as dfl

#only need to run first time
#try:
#    _create_unverified_https_context = ssl._create_unverified_context
#except AttributeError:
#    pass
#else:
#    ssl._create_default_https_context = _create_unverified_https_context

#nltk.download("vader_lexicon") # downloads vader_lexicon to /Users/nbachman/nltk_data
#print("import nltk,vader and SentimentIntesity complete")

#create a simple Lexicon to determine Democrat or Republican context
R_Lexicon = ["Republican","Republicans","Republican's","GOP", "G.O.P","Conservative","Conservatives","Trump","Trump's","Pentz","Ryan","McConnell","Bush","McCarthy", "Scalise", "Boehner","Romney","Cruz","Rubio","Cain"]
D_Lexicon = ["Democrat","Democrats","Democrat's","Democratic","Liberal","Liberals","Clinton","Clinton's","Biden","Pelosi","Schumer","Reid","Hoyer","Clyburn","Obama","Harris","Kaine","Sanders"]

from nltk.sentiment.vader import SentimentIntensityAnalyzer
#update the VDER Lexicon to put more weight on popular political terms
NEG = -3
POS = 3
NET = 0

N_Lexicon = {u'landslide': NEG,u'delusional': NEG,u'hackers': NEG,u'hate': NEG,u'terror': NEG,u'threat': NEG,u'fear': NEG,u'destruction': NEG,u'danger': NEG, u'murder': NEG, u'enemies': NEG, u'hard': NEG, u'confront': NEG, u'threats': NEG, u'terrorism': NEG, u'problems': NEG, u'difficult': NEG, u'conflict': NEG, u'tyranny': NEG, u'evil': NEG, u'terriable': NEG, u'oppression': NEG, u'hatred': NEG, u'attack': NEG}
P_Lexicon = {u'approve': POS,u'unify': POS, u'pass': POS, u'freedom': POS, u'work': POS, u'peace': POS, u'good': POS, u'free': POS, u'liberty': POS, u'thank': POS, u'reform': POS, u'support': POS, u'better': POS, u'progress': POS, u'commitment': POS, u'dignity': POS, u'important': POS, u'courage': POS, u'well': POS,  u'strong': POS,  u'protect': POS,  u'honor': POS}
Z_Lexicon = {u'party': NET}

#Create an instance of SentimentIntesityAnalyzer called vader
vader = SentimentIntensityAnalyzer() #

#Test a sample of text
#sample = "This Is a Jobs Report That Democrats Can Boast About"
#print (sample)
#print (vader.polarity_scores(sample))

#Adds terms and weights to the VADAR
vader.lexicon.update(N_Lexicon)
vader.lexicon.update(P_Lexicon)

#Funtion that determines if the context is either Republican, Democrat or None
def get_context(dataframe):
    results = []
    for Title in dataframe: 
        if any(ele in Title for ele in R_Lexicon):
            results.append("Republican")
        elif any(ele in Title for ele in D_Lexicon):
            results.append("Democrat")
        else:
            results.append("None")
    return results

#Funtion that determines if the articale is Bias to left, right or netutral
def get_bias(df):
    results = []
    for index, row in df.iterrows():
        if (row["sentiment"] == "negative" and row["context"] == "Democrat"):
            row["Bias"]  = "Right"
        elif (row["sentiment"] == "positive" and row["context"] == "Democrat"):
            row["Bias"]  = "Left"
        elif row["sentiment"] == "positive" and row["context"] == "Republican":
            row["Bias"] = "Right"
        elif row["sentiment"] == "negative" and row["context"] == "Republican":
            row["Bias"]  = "Left"
        else:
            row["Bias"]  = "Neutral"
        results.append(row["Bias"])
    return results

#Calculate the number of accurate matches and accuracy
def get_accuracy(df):
    #Find out if Predicted_Bias matches labeled Bias 
    df['matches'] =df.apply(lambda row: row["Bias"] in row["Predicted_Bias"],axis=1)

    Correct_Predictions = df['matches'].values.sum()
    Accuracy = Correct_Predictions / len(df)
    print("Accuracy", Accuracy)
    return df

#Initialize Pandas Dataframe with test data
df = pd.read_csv("/Users/nbachman/Documents/HCP Anywhere/GradSchool/Text Mining and Analytics/CS410-BiasDetector/data_labeled/biasdetective3.csv")

#Add a sentiment score of the Title column to dataframe 
df["scores"] = df["Title"].apply(lambda Title: vader.polarity_scores(Title))

#Add a column with just the compound value
df["compound"] = df["scores"].apply(lambda score_dict: score_dict["compound"])

#Add a sentiment column for pos, neg or neutral
df["sentiment"] = df["compound"].apply(lambda c: "positive" if c > .02 else ("negative" if c < -.02 else "neutral"))

#Add a Context Title column to dataframe 
df["context"] = get_context(df["Title"])

#Add a Bias Title column to dataframe 
df["Predicted_Bias"] = get_bias(df)

#Calculate the number of accurate matches and accuracy
get_accuracy(df)

#Export Results to CSV file
df.to_csv("/Users/nbachman/Documents/HCP Anywhere/GradSchool/Text Mining and Analytics/CS410-BiasDetector/DT_Sentiment.csv")