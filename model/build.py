import pandas as pd
from user_agents import parse

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, make_union
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


class PandasSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, records=False):
        self.columns = columns
        self.records = records

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.records:
            return X[self.columns].to_dict(orient="records")
        return X[self.columns]


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y):
        X = X.assign(target=y)
        self.frequencies = X.groupby(self.cols).size() / X.shape[0]
        return self

    def transform(self, X):
        transformed = X[self.cols].map(self.frequencies)
        # nans correspond to zero frequency
        return transformed.fillna(0.0).values.reshape(-1, 1)


class UserAgentParser(BaseEstimator, TransformerMixin):
    def __init__(self, cols, ocols):
        self.cols = cols
        self.ocols = ocols

    def fit(self, X, y):
        return self

    def transform(self, X):
        parsed = X[self.cols].apply(self._parse)

        out = pd.DataFrame({
            col: parsed.str[i] for i, col in enumerate(self.ocols)
        })
        return out

    def _parse(self, ua):
        # It's not the fastest way to do it, but it's clean
        user_agent = parse(ua)
        return str(user_agent).split(" / ")


def preprocess_url(url):
    return " ".join(url.split("/"))


def build_model():
    model = make_pipeline(
        # Prepare the data
        make_union(
            make_pipeline(
                PandasSelector("url"),
                TfidfVectorizer(
                    preprocessor=preprocess_url,
                    min_df=10,  # can be used to regularize the model
                    # max_df=1.0,  # can be used to regularize the model
                )
            ),
            make_pipeline(
                PandasSelector(["city"]),
                FrequencyEncoder("city"),
                StandardScaler(),
            ),
            make_pipeline(
                PandasSelector(["ua"]),
                # It's not effective to add it here, it doesn't learn
                # anything, so it can be applied outside the pipeline
                # but this way it's easier to experiment
                UserAgentParser("ua", ocols=["device", "os", "browser"]),
                make_union(
                    FrequencyEncoder("device"),
                    FrequencyEncoder("os"),
                    FrequencyEncoder("browser"),
                ),
                StandardScaler(),
            ),
        ),

        # Trained model

        # By default this fits one-vs-rest with n_classes of classifications
        # Here we want multinomial loss, for the problem
        LogisticRegression(max_iter=1000, multi_class='multinomial'),
    )
    return model
