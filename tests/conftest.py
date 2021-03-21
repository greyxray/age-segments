import pytest
import pandas as pd


def dummy_users(idx, provider="provider1"):
    users = [
        {
            'user': idx + 1, 'provider': provider, 'age': '60+',
            'url': 'www.google.com', 'city': 'bobruisk',
            'ua': 'Opera/9.80 (Windows NT 6.0) Presto/2 Version/12',
        },
        {
            'user': idx + 2, 'provider': provider, 'age': '50-59',
            'url': 'www.yahoo.com', 'city': 'paris',
            'ua': 'Opera/9.80 (Windows NT 6.0) Presto/2 Version/12',
        },
        {
            'user': idx + 3, 'provider': provider, 'age': '30-39',
            'url': 'www.bing.com', 'city': 'london',
            'ua': 'Opera/9.80 (Windows NT 6.0) Presto/2 Version/12',
        },
        {
            'user': idx + 4, 'provider': provider, 'age': '40-49',
            'url': 'www.duckduckgo.com', 'city': 'berlin',
            'ua': 'Opera/9.80 (Windows NT 6.0) Presto/2 Version/12',
        },
        {
            'user': idx + 5, 'provider': provider, 'age': '20-29',
            'url': 'www.bing.com', 'city': 'rome',
            'ua': 'Opera/9.80 (Windows NT 6.0) Presto/2 Version/12',
        },
    ]
    return users


@pytest.fixture()
def data():
    dataset = []
    for i in range(100):
        dataset += dummy_users(i)
    return pd.DataFrame(dataset)
