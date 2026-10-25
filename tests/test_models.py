import numpy as np
import pandas as pd
import pytest

import config
from models import (build_dial_portfolios, growth_anchor, min_variance, risk_contributions, risk_parity,
                    weights_for_level)

FIVE_ASSETS = ["A", "B", "C", "D", "E"]


def make_cov(tickers, seed):
    # random positive-definite daily covariance matrix of realistic size
    rng = np.random.default_rng(seed)
    n = len(tickers)
    loadings = rng.normal(size=(n, n))
    cov = (loadings @ loadings.T / n + 0.5 * np.eye(n)) * 1e-4
    return pd.DataFrame(cov, index=tickers, columns=tickers)


def test_risk_contributions_sum_to_one():
    cov = make_cov(FIVE_ASSETS, seed=0)
    weights = pd.Series([0.1, 0.2, 0.3, 0.15, 0.25], index=FIVE_ASSETS)
    contributions = risk_contributions(weights, cov)
    assert contributions.sum() == pytest.approx(1.0, abs=1e-12)


def test_risk_contributions_equal_for_uncorrelated_equal_variance_assets():
    cov = pd.DataFrame(1e-4 * np.eye(5), index=FIVE_ASSETS, columns=FIVE_ASSETS)
    weights = pd.Series(0.2, index=FIVE_ASSETS)
    contributions = risk_contributions(weights, cov)
    assert np.allclose(contributions.values, 1.0 / 5)


def test_min_variance_weights_and_variance():
    cov = make_cov(FIVE_ASSETS, seed=1)
    weights, success = min_variance(cov, config.MAX_WEIGHT)
    equal = np.full(5, 1.0 / 5)
    assert success
    assert weights.sum() == pytest.approx(1.0, abs=1e-8)
    assert (weights >= -1e-8).all()
    assert (weights <= config.MAX_WEIGHT + 1e-8).all()
    assert weights.values @ cov.values @ weights.values <= equal @ cov.values @ equal


def test_risk_parity_equalizes_risk_contributions():
    cov = make_cov(FIVE_ASSETS, seed=2)
    weights, success = risk_parity(cov)
    contributions = risk_contributions(weights, cov)
    assert success
    assert np.abs(contributions.values - 1.0 / 5).max() < 1e-4


def test_weights_for_level_anchors_and_range():
    cov = make_cov(config.TICKERS, seed=3)
    conservative, success = min_variance(cov, config.MAX_WEIGHT)
    growth = growth_anchor(cov.index)
    assert success
    assert np.allclose(weights_for_level(1, conservative, growth).values, conservative.values)
    assert np.allclose(weights_for_level(5, conservative, growth).values, growth.values)
    dial = build_dial_portfolios(cov, [1, 5])
    assert np.allclose(dial[1].values, conservative.values)
    assert np.allclose(dial[5].values, growth.values)
    for bad_level in [0, 6, 0.99, 5.01]:
        with pytest.raises(ValueError):
            weights_for_level(bad_level, conservative, growth)
