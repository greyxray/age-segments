import click
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from model.build import build_model


@click.command()
@click.option("--data", type=click.Path(exists=True),
              default="data/data.csv")
def main(data):
    df = pd.read_csv(data).fillna("UNK")

    provider1 = df[df["provider"] == "provider1"]
    provider2 = df[df["provider"] == "provider2"]

    X_tr, X_te, y_tr, y_te = train_test_split(provider1, provider1["age"])
    model = build_model()
    model.fit(X_tr, y_tr)

    print("Done fitting")

    print("Train set:")
    print(classification_report(y_tr, model.predict(X_tr)))

    print("Valid set:")
    print(classification_report(y_te, model.predict(X_te)))

    print("Hold-out (provider2) set:")
    print(classification_report(provider2["age"], model.predict(provider2)))


if __name__ == '__main__':
    main()
