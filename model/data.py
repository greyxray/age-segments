import click
import pandas as pd


@click.command()
@click.option("--data", type=click.Path(exists=True),
              default="data/data.csv")
def main(data):
    df = pd.read_csv(data)
    print(df.head())
    import ipdb; ipdb.set_trace(); import IPython; IPython.embed() # noqa


if __name__ == '__main__':
    main()
