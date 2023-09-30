# %%

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np


# %%
def model(data):
    keep_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "subjectivity",
        "polarity",
        "compound",
        "negative",
        "neutral",
        "positive",
        "label",
    ]

    data = data[keep_columns]

    model = LinearDiscriminantAnalysis()

    x = np.array(data.drop(["label"], axis=1))
    y = np.array(data["label"])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    report = classification_report(y_test, predictions)

    return report


# %%
