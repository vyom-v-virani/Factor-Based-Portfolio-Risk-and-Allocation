import numpy as np
import pandas as pd

import config
from data import train_test_split


def make_dataset(n_days):
    # small seeded dataset with the same three frames as build_dataset()
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    returns = pd.DataFrame(rng.normal(0, 0.01, size=(n_days, 3)), index=dates, columns=["A", "B", "C"])
    factors = pd.DataFrame(rng.normal(0, 0.01, size=(n_days, 4)), index=dates, columns=["Mkt-RF", "SMB", "HML", "RF"])
    excess_returns = returns.sub(factors["RF"], axis=0)
    return {"returns": returns, "excess_returns": excess_returns, "factors": factors}


def test_train_dates_are_all_before_test_dates():
    dataset = make_dataset(200)
    train, test = train_test_split(dataset)
    for name in ["returns", "excess_returns", "factors"]:
        assert train[name].index.max() < test[name].index.min()
        assert (train[name].index.values[:, None] < test[name].index.values[None, :]).all()
        assert len(train[name]) + len(test[name]) == 200
    assert len(train["returns"]) == int(200 * config.TRAIN_FRAC)
