


# %%
def horizon_analysis(data):
    horizons = [2, 5, 60, 250, 1000]

    def indicators(data):
        for horizon in horizons:
            rolling_avg = data.loc[:,"Close"].rolling(horizon).mean()

            ratio_column = f"Close_Ratio_{horizon}"
            data.loc[:,ratio_column] = data.loc[:,"Close"] / rolling_avg

            trend_column = f"Trend_{horizon}"
            data.loc[:,trend_column] = data.loc[:,"label"].shift(1).rolling(horizon).sum()

    indicators(data)
    data = data.dropna()

    return data


# %%
