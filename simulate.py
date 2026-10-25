import numpy as np
import pandas as pd

import config


def reconstruct_returns(models, factors):
    # excess returns = factors @ betas + residuals, with alpha left out
    betas = models["params"][config.FACTOR_COLS]
    systematic = factors[config.FACTOR_COLS] @ betas.T
    return systematic[models["residuals"].columns] + models["residuals"]


def block_bootstrap_indices(n_obs, n_paths, horizon, block, rng):
    # circular block bootstrap: row positions of shape (n_paths, horizon)
    n_blocks = -(-horizon // block)
    starts = rng.integers(0, n_obs, size=(n_paths, n_blocks))
    positions = (starts[:, :, None] + np.arange(block)[None, None, :]) % n_obs
    return positions.reshape(n_paths, n_blocks * block)[:, :horizon]


def simulate_portfolios(returns_ex_alpha, rf, weights_df, seed):
    # bootstrap compounded 1-year returns and pooled 1-day returns for each portfolio (daily rebalanced)
    weights = weights_df.reindex(returns_ex_alpha.columns).values
    total = returns_ex_alpha.values @ weights + rf.values[:, None]  # same dates give excess return and RF
    rng = np.random.default_rng(seed)
    positions = block_bootstrap_indices(len(total), config.N_PATHS, config.HORIZON_DAYS, config.BLOCK_DAYS, rng)
    daily = total[positions]  # (paths, days, portfolios)
    horizon = np.prod(1.0 + daily, axis=1) - 1.0
    one_day = daily.reshape(-1, daily.shape[2])
    return {"horizon": pd.DataFrame(horizon, columns=weights_df.columns),
            "one_day": pd.DataFrame(one_day, columns=weights_df.columns)}


def risk_stats(returns):
    # mean, median, VaR and CVaR (positive losses), probability of loss and worst outcome per portfolio
    values = returns.values
    stats = {"mean": values.mean(axis=0), "median": np.median(values, axis=0)}
    for level in config.VAR_LEVELS:
        tag = int(round(level * 100))
        quantile = np.quantile(values, 1.0 - level, axis=0)
        cvar = []
        for i in range(values.shape[1]):
            tail = values[:, i][values[:, i] <= quantile[i]]
            cvar.append(-tail.mean())
        stats[f"var_{tag}"] = -quantile
        stats[f"cvar_{tag}"] = np.array(cvar)
    stats["prob_loss"] = (values < 0).mean(axis=0)
    stats["worst"] = values.min(axis=0)
    return pd.DataFrame(stats, index=returns.columns)


def factor_exposures(weights_df, params):
    # portfolio betas = weights @ betas (portfolio x factor)
    return weights_df.T @ params[config.FACTOR_COLS]


def market_stress(exposures):
    # linear one-factor approximation: market beta times the shock (ignores SMB and HML)
    stress = {}
    for shock in config.STRESS_MARKET_SHOCKS:
        stress[shock] = exposures[config.FACTOR_COLS[0]] * shock
    return pd.DataFrame(stress)


def scenario_replay(daily_returns, scenarios):
    # cumulative return of each portfolio over each named date range
    replay = {}
    for name in scenarios:
        start, end = scenarios[name]
        replay[name] = (1.0 + daily_returns.loc[start:end]).prod() - 1.0
    return pd.DataFrame(replay)
