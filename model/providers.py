import click
import pandas as pd
import matplotlib.pyplot as plt


'''
Check which provider to use
'''


def hist(x, label, c="b"):
    counts = x.value_counts(normalize=False)
    counts.sort_index().plot(kind='bar', alpha=0.1, label=label, color=c)


def top(x, n=20):
    counts = x.value_counts(normalize=False)
    return counts.sort_values(ascending=False)[:20]


@click.command()
@click.option("--data", type=click.Path(exists=True),
              default="data/data.csv")

def main(data):
    df = pd.read_csv(data)
    hist(df["age"], label="all")

    provider1 = df.loc[df["provider"] == "provider1"]
    hist(provider1["age"], label="provider1", c="r")

    provider2 = df.loc[df["provider"] == "provider2"]
    hist(provider2["age"], label="provider2", c="g")

    print("All providers:")
    print(top(df["city"]))

    print("Provider 1:")
    print(top(provider1["city"]))

    print("Provider 2:")
    print(top(provider2["city"]))

    provider1_cities = set(provider1.city.unique())
    provider2_cities = set(provider2.city.unique())
    print("Cities covered by both providers:",
        len(provider1_cities.intersection(provider2_cities)))
    print("Cities only in provider1:", len(provider1_cities - provider2_cities))
    print("Cities only in provider2:", len(provider2_cities - provider1_cities))

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
