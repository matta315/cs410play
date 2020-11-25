import nltk
import ssl
import numpy as np
import pandas as pd

#only need to run first time
#try:
#    _create_unverified_https_category = ssl._create_unverified_category
#except AttributeError:
#    pass
#else:
#    ssl._create_default_https_category = _create_unverified_https_category

#nltk.download("vader_lexicon") # downloads vader_lexicon to /Users/nbachman/nltk_data

#create a simple Lexicon to determine Democrat or Republican category
R_Lexicon = ["republican","republicans","republican's","gop", "g.o.p","conservative","conservatives","trump","trump's","pentz","ryan","mcconnell","bush","mccarthy", "scalise", "boehner","romney","cruz","rubio","cain","giuliani"]
D_Lexicon = ["democrat","democrats","democrat's","democratic","liberal","liberals","clinton","clinton's","biden","biden's","pelosi","schumer","reid","hoyer","clyburn","obama","harris","kaine","sanders","feinstein"]

#Import sentiment analyzer from VADER 
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#Update the VADER Lexicon to put more weight on popular politicaly charged terms
NEG = -3
POS = 3
NET = 0

NEG_Lexicon = {u'losers': NEG, u'lose': NEG,u'impeached': NEG,u'landslide': NEG,u'repeal': NEG,u'replace': NEG,u'delusional': NEG,u'hackers': NEG, u'hacking': NEG, u'hate': NEG,u'terror': NEG,u'threat': NEG,u'kill': NEG,u'fear': NEG,u'destruction': NEG,u'danger': NEG, u'murder': NEG, u'enemies': NEG, u'hard': NEG, u'war': NEG, u'confront': NEG, u'threats': NEG, u'terrorism': NEG, u'problems': NEG, u'difficult': NEG, u'conflict': NEG, u'tyranny': NEG, u'evil': NEG, u'terriable': NEG, u'oppression': NEG, u'hatred': NEG, u'attack': NEG,u'fraud': NEG}
POS_Lexicon = {u'unity': POS, u'win': POS, u'winners': POS, u'pass': POS, u'freedom': POS, u'work': POS, u'peace': POS, u'good': POS, u'free': POS, u'liberty': POS, u'thank': POS, u'reform': POS, u'support': POS, u'better': POS, u'progress': POS, u'commitment': POS, u'dignity': POS, u'important': POS, u'courage': POS, u'well': POS,  u'strong': POS,  u'protect': POS,  u'honor': POS}
NET_Lexicon = {u'party': NET}

#Create an instance of SentimentIntesityAnalyzer called vader
vader = SentimentIntensityAnalyzer() #

#Test a sample of text
#sample = "This Is a Jobs Report That Democrats Can Boast About"
#print (sample)
#print (vader.polarity_scores(sample))

#Adds terms and weights to the VADAR
vader.lexicon.update(NEG_Lexicon)
vader.lexicon.update(POS_Lexicon)
vader.lexicon.update(NET_Lexicon)

#Funtion that determines if the category is either Republican, Democrat or None
def get_category(dataframe):
    results = []
    for title in dataframe: 
        if any(ele in title for ele in R_Lexicon):
            results.append("Republican")
        elif any(ele in title for ele in D_Lexicon):
            results.append("Democrat")
        else:
            results.append("None")
    return results

#Funtion that determines if the artical is bias to left, right or netutral
def get_bias(df):
    results = []
    for index, row in df.iterrows():
        if (row["sentiment"] == "negative" and row["category"] == "Democrat"):
            row["bias"]  = "Right"
        elif (row["sentiment"] == "positive" and row["category"] == "Democrat"):
            row["bias"]  = "Left"
        elif row["sentiment"] == "positive" and row["category"] == "Republican":
            row["bias"] = "Right"
        elif row["sentiment"] == "negative" and row["category"] == "Republican":
            row["bias"]  = "Left"
        else:
            row["bias"]  = "Neutral"
        results.append(row["bias"])
    return results

if __name__ == "__main__":
    #Initialize Pandas Dataframe with test data
    df = pd.read_csv("/Users/nbachman/Documents/HCP Anywhere/GradSchool/Text Mining and Analytics/CS410-BiasDetector/data_raw/NPR_All.csv")

    #Remove any extra quotes from the titles for better matching
    df["title"] = df["title"].apply(lambda x: x.replace('"', ''))
    df["title"] = df["title"].apply(lambda x: x.replace("'", ''))
    df["title"] = df["title"].apply(lambda x: x.lower())

    #Add a sentiment score column to dataframe 
    df["scores"] = df["title"].apply(lambda title: vader.polarity_scores(title))

    #Add compound value to the dataframe
    df["compound"] = df["scores"].apply(lambda score_dict: score_dict["compound"])

    #Add a new sentiment column for pos, neg or neutral
    df["sentiment"] = df["compound"].apply(lambda c: "positive" if c > .02 else ("negative" if c < -.02 else "neutral"))

    #Add a category title column to dataframe 
    df["category"] = get_category(df["title"])

    #Add a Bias title column to dataframe 
    df["bias"] = get_bias(df)

    #Print the number of neutral, positve and negative headlines
    print (df["bias"].value_counts())

     #Print the number of neutral, positve and negative headlines
    print (df["sentiment"].value_counts())

    #Export Results to CSV file
    df.to_csv("/Users/nbachman/Documents/HCP Anywhere/GradSchool/Text Mining and Analytics/CS410-BiasDetector/test.csv")