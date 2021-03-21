import click
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
# from imblearn.over_sampling import RandomOverSampler

from model.build import build_model


'''
Reformulate the problem as a binary classification
to target 20-29 group
'''


RANDOM_SEED = 137
np.random.seed(RANDOM_SEED)


NEWAGE = {
    "20-29": "20-29",
    "30-39": "rest",
    "40-49": "rest",
    "50-59": "rest",
    "60+": "rest",
}


@click.command()
@click.option("--data", type=click.Path(exists=True),
              default="data/data.csv")
def main(data):
    df = pd.read_csv(data).fillna("UNK")
    df["age*"] = df["age"].map(NEWAGE)

    provider1 = df[df["provider"] == "provider1"]
    provider2 = df[df["provider"] == "provider2"]

    X_tr, X_te, y_tr, y_te = train_test_split(provider1, provider1["age*"])

    rs = RandomUnderSampler(random_state=0)
    X_tr_resampled, y_tr_resampled = rs.fit_resample(X_tr, y_tr)

    model = build_model()
    model.fit(X_tr_resampled, y_tr_resampled)
    # model.fit(X_tr, y_tr)

    print("Done fitting")

    print("Train set")
    print(classification_report(y_tr, model.predict(X_tr)))

    print("Valid set")
    print(classification_report(y_te, model.predict(X_te)))

    print("Hold-out set")
    print(classification_report(provider2["age*"], model.predict(provider2)))


if __name__ == '__main__':
    main()
