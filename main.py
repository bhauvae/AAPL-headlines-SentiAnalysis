# %%
from data import data
from machine_learning import model
from technical_analysis import horizon_analysis
from sentiment_analysis import sentiment_analysis



if __name__ == "__main__":
    # without horizon analysis
    report_only_sentiment = model(sentiment_analysis(data))
    # with horizon analysis
    # report_with_horizon_analysis = model(sentiment_analysis(horizon_analysis(data)))
    pass
# %%
