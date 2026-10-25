import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

import config


def fit_factor_models(excess_returns, factors):
    # OLS of each ticker's excess return on the factors, with Newey-West (HAC) standard errors
    X = sm.add_constant(factors[config.FACTOR_COLS], has_constant="add")
    X = X.rename(columns={"const": "alpha"})
    params = {}
    tvalues = {}
    r2 = {}
    residuals = {}
    for ticker in excess_returns.columns:
        fit = sm.OLS(excess_returns[ticker], X).fit(cov_type="HAC", cov_kwds={"maxlags": config.HAC_LAGS})
        params[ticker] = fit.params
        tvalues[ticker] = fit.tvalues
        r2[ticker] = fit.rsquared
        residuals[ticker] = fit.resid
    columns = ["alpha"] + config.FACTOR_COLS
    return {"params": pd.DataFrame(params).T[columns], "tvalues": pd.DataFrame(tvalues).T[columns],
            "r2": pd.Series(r2, name="R2"), "residuals": pd.DataFrame(residuals)}


def ledoit_wolf_cov(returns):
    # Ledoit-Wolf shrunk daily covariance; returns the covariance DataFrame and the shrinkage
    lw = LedoitWolf().fit(returns.values)
    cov = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
    return cov, lw.shrinkage_


def equal_weight(tickers):
    # 1/N weights
    return pd.Series(1.0 / len(tickers), index=list(tickers))


def risk_contributions(weights, cov):
    # fractional risk contribution of each asset: w_i * (cov w)_i / (w' cov w)
    w = weights.values
    marginal = cov.values @ w
    return pd.Series(w * marginal / (w @ marginal), index=cov.index)


def portfolio_vol(weights, cov):
    # annualized portfolio volatility
    w = weights.values
    return np.sqrt(w @ cov.values @ w * config.TRADING_DAYS)


def sum_to_one(w):
    return w.sum() - 1.0


def min_variance_objective(w, S, scale):
    return scale * (w @ S @ w)


def risk_parity_objective(w, S, target):
    marginal = S @ w
    rc = w * marginal / (w @ marginal)
    return np.sum((rc - target) ** 2) * len(w) ** 2


def min_variance(cov, max_weight):
    # long-only minimum variance with a per-asset cap; returns (weights, optimizer success)
    n = len(cov)
    S = cov.values
    scale = n / np.trace(S)  # daily variances are tiny, so rescale the objective
    result = minimize(min_variance_objective, np.full(n, 1.0 / n), args=(S, scale), method="SLSQP",
                      bounds=[(0.0, max_weight)] * n, constraints=[{"type": "eq", "fun": sum_to_one}],
                      options={"ftol": 1e-10, "maxiter": 1000})
    return pd.Series(result.x, index=cov.index), result.success


def risk_parity(cov):
    # long-only risk parity (no cap), starting from inverse-volatility weights; returns (weights, success)
    n = len(cov)
    S = cov.values
    inverse_vol = 1.0 / np.sqrt(np.diag(S))
    start = inverse_vol / inverse_vol.sum()
    result = minimize(risk_parity_objective, start, args=(S, 1.0 / n), method="SLSQP",
                      bounds=[(1e-6, 1.0)] * n, constraints=[{"type": "eq", "fun": sum_to_one}],
                      options={"ftol": 1e-10, "maxiter": 1000})
    return pd.Series(result.x, index=cov.index), result.success


def add_ratio(table):
    # risk_contribution / weight, left as NaN where the weight is below 1e-6
    ratio = table["risk_contribution"] / table["weight"]
    table["risk_to_weight_ratio"] = ratio.where(table["weight"] >= 1e-6, np.nan)
    return table


def contribution_table(weights, cov):
    # per-ticker weight, risk contribution and risk-to-weight ratio
    table = pd.DataFrame({"weight": weights.reindex(cov.index), "risk_contribution": risk_contributions(weights, cov)})
    return add_ratio(table)


def asset_class_summary(weights, cov):
    # weight, risk contribution and ratio summed within each asset class
    table = contribution_table(weights, cov)
    table["asset_class"] = table.index.map(config.ASSET_CLASS)
    summary = table.groupby("asset_class")[["weight", "risk_contribution"]].sum()
    summary = summary.reindex(config.ASSET_CLASS_ORDER).fillna(0.0)
    return add_ratio(summary)


def growth_anchor(tickers):
    # equal weight across the equity tickers, zero elsewhere
    weights = pd.Series(0.0, index=tickers)
    weights[config.EQUITY_TICKERS] = 1.0 / len(config.EQUITY_TICKERS)
    return weights


def weights_for_level(level, conservative, growth):
    # blend of the conservative and growth anchors: a = (level - 1) / 4
    low = min(config.RISK_LEVELS)
    high = max(config.RISK_LEVELS)
    if level < low or level > high:
        raise ValueError(f"Risk level must be a number in [{low}, {high}]; got {level}.")
    a = (level - low) / (high - low)
    return (1 - a) * conservative + a * growth


def build_dial_portfolios(cov, levels):
    # weights for each risk level, computed from the covariance matrix alone
    conservative, success = min_variance(cov, config.MAX_WEIGHT)
    if not success:
        raise ValueError("Min-variance optimization failed.")
    growth = growth_anchor(cov.index)
    portfolios = {}
    for level in levels:
        portfolios[level] = weights_for_level(level, conservative, growth)
    return portfolios


def dial_profile(level, cov, weights):
    # weights, volatility and asset-class mix and risk for one dial level
    nearest = config.RISK_LEVELS[0]
    for risk_level in config.RISK_LEVELS:
        if abs(risk_level - level) < abs(nearest - level):
            nearest = risk_level
    summary = asset_class_summary(weights, cov)
    return {"level": level, "label": config.RISK_LABELS[nearest], "weights": weights,
            "volatility": portfolio_vol(weights, cov),
            "class_mix": summary["weight"], "class_risk": summary["risk_contribution"]}
