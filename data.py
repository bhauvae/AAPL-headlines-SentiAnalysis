# %%
import pandas as pd

# %%
data = pd.read_csv(r"data\AppleNewsStock.csv")
data.set_index("Date", inplace=True)
# %%
data["Next Day"] = data["Close"].shift(-1)
data["label"] = (data["Next Day"] >= data["Close"]).astype(int)
del data["Next Day"]
