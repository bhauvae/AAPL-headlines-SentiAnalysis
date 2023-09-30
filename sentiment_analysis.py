# %%
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# %%
def sentiment_analysis(data):
    data = data.dropna()

    def getSubjectivity(text):
        return TextBlob(text).sentiment.subjectivity

    def getPolarity(text):
        return TextBlob(text).sentiment.polarity

    def getSIA_scores(text):
        return SentimentIntensityAnalyzer().polarity_scores(text)

    data.loc[:, "subjectivity"] = data.loc[:, "News"].apply(getSubjectivity)
    data.loc[:, "polarity"] = data.loc[:, "News"].apply(getPolarity)

    data.loc[:, "SIA_scores"] = data.loc[:, "News"].apply(getSIA_scores)
    data.loc[:, "compound"] = data.loc[:, "SIA_scores"].apply(lambda x: x["compound"])
    data.loc[:, "negative"] = data.loc[:, "SIA_scores"].apply(lambda x: x["neg"])
    data.loc[:, "neutral"] = data.loc[:, "SIA_scores"].apply(lambda x: x["neu"])
    data.loc[:, "positive"] = data.loc[:, "SIA_scores"].apply(lambda x: x["pos"])
    labels = data["label"].copy()
    del data["label"]
    del data["SIA_scores"]
    del data["News"]
    data["label"] = labels


    return data


# %%
