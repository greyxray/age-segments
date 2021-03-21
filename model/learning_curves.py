import click
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve

from model.build import build_model

sns.set_style()


'''
Plot score learning curve
'''


def curve(x, y, label, color):
    mean = y.mean(axis=1)
    plt.plot(x, mean, 'o-', label=label)

    std = y.std(axis=1)
    plt.fill_between(x, mean - std, mean + std, alpha=0.1)


@click.command()
@click.option("--data", type=click.Path(exists=True),
              default="data/data.csv")
def main(data):
    df = pd.read_csv(data).fillna("UNK")
    provider1 = df[df["provider"] == "provider1"]

    X_tr, X_te, y_tr, y_te = train_test_split(provider1, provider1["age"])
    model = build_model()

    train_sizes, train_scores, test_scores = learning_curve(
        model, X_tr, y_tr,
        cv=2,
        scoring="f1_micro",
        n_jobs=-1,
    )

    curve(train_sizes, train_scores, label="train", color="b")
    curve(train_sizes, test_scores, label="test", color="orange")

    plt.ylabel("f1 score")
    plt.xlabel("Number of training samples")
    plt.ylim(0.2, 0.6)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
