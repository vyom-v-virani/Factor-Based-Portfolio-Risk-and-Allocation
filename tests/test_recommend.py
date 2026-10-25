import numpy as np
import pandas as pd
import pytest

import config
from recommend import allocation_as_of, previous_month_end, trailing_window


def make_returns(n_days, seed):
    # seeded synthetic daily returns for the configured tickers on business days
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    data = rng.normal(0.0003, 0.01, size=(n_days, len(config.TICKERS)))
    return pd.DataFrame(data, index=dates, columns=config.TICKERS)


def first_friday_after(returns, position):
    # position of the first Friday at or after the given position
    for i in range(position, len(returns)):
        if returns.index[i].dayofweek == 4:
            return i
    raise ValueError("no Friday found")


def test_as_of_truncation_ignores_later_data():
    returns = make_returns(1000, seed=0)
    position = first_friday_after(returns, 900)
    asof = returns.index[position] + pd.Timedelta(days=1)  # a Saturday, so the Friday is the as-of trading day
    weights, cov, window = allocation_as_of(returns, asof, 3)

    changed = returns.copy()
    rng = np.random.default_rng(1)
    changed.iloc[position + 1:] = rng.normal(0.0, 0.05, size=changed.iloc[position + 1:].shape)
    weights_changed, cov_changed, window_changed = allocation_as_of(changed, asof, 3)

    assert window.index.max() == returns.index[position]
    assert len(window) == config.WINDOW_DAYS
    # the window is identical; the covariance can differ in the last bit (array layout), which the optimizer
    # turns into ~1e-10 differences in the weights, so compare with the project's 1e-8 weight tolerance
    assert window.equals(window_changed)
    assert np.allclose(cov.values, cov_changed.values, rtol=0, atol=1e-15)
    assert np.allclose(weights.values, weights_changed.values, rtol=0, atol=1e-8)


def test_window_needs_enough_history():
    returns = make_returns(1000, seed=0)
    with pytest.raises(ValueError):
        trailing_window(returns, returns.index[config.WINDOW_DAYS - 2])


def test_previous_month_end_is_in_an_earlier_month():
    returns = make_returns(1000, seed=0)
    asof_date = returns.index[900]
    previous = previous_month_end(returns, asof_date)
    assert previous < asof_date
    assert previous.to_period("M") < asof_date.to_period("M")
    month_days = returns.index[returns.index.to_period("M") == previous.to_period("M")]
    assert previous == month_days.max()


def test_invalid_level_raises_the_existing_error():
    returns = make_returns(1000, seed=0)
    asof = returns.index[900]
    for bad_level in [0, 6, 5.01]:
        with pytest.raises(ValueError, match="Risk level must be a number in"):
            allocation_as_of(returns, asof, bad_level)
