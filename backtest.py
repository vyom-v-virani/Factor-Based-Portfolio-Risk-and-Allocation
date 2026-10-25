import numpy as np
import pandas as pd

import config
from models import build_dial_portfolios, equal_weight, ledoit_wolf_cov, risk_parity


def target_weights(strategy, cov, tickers):
    # target weights (in ticker order) for one strategy given the rolling covariance
    kind = strategy["kind"]
    if kind == "dial":
        weights = build_dial_portfolios(cov, [strategy["level"]])[strategy["level"]]
    elif kind == "equal":
        weights = equal_weight(tickers)
    elif kind == "parity":
        weights, success = risk_parity(cov)
        if not success:
            raise ValueError("Risk parity optimization failed.")
    else:  # "fixed" and "buyhold" use the weights given in the strategy
        weights = pd.Series(strategy["weights"]).reindex(tickers).fillna(0.0)
    return weights.values


def net_returns(result, cost_bps):
    # gross returns minus cost_bps * traded amount, taken on the first day of each new holding period
    traded = result["traded"].copy()
    traded.index = pd.DatetimeIndex(result["schedule"]["apply_start"])
    cost = traded.reindex(result["gross"].index).fillna(0.0) * cost_bps / 10000
    return result["gross"] - cost


def walk_forward(returns, rf, tickers, strategies, cost_bps):
    # monthly walk-forward backtest: weights from the trailing window apply from the next trading day on
    window = config.WINDOW_DAYS
    data = returns[tickers]
    values = data.to_numpy()
    index = data.index
    n_days = len(index)

    # last trading day of each month, kept if the window is full and a next day exists
    last_of_month = pd.Series(np.arange(n_days), index=index).groupby(index.to_period(config.REBALANCE_FREQ)).max()
    positions = []
    for position in last_of_month:
        if position >= window - 1 and position < n_days - 1:
            positions.append(position)
    last_day = positions[1:] + [n_days - 1]  # last day of each holding period

    # targets at each rebalance, using only data up to and including that day
    targets = {}
    for strategy in strategies:
        targets[strategy["name"]] = []
    schedule_rows = []
    for position in positions:
        window_returns = data.iloc[position - window + 1:position + 1]
        cov, shrinkage = ledoit_wolf_cov(window_returns)
        for strategy in strategies:
            targets[strategy["name"]].append(target_weights(strategy, cov, tickers))
        schedule_rows.append({"rebalance_date": index[position], "window_start": window_returns.index[0],
                              "window_end": window_returns.index[-1], "n_obs": len(window_returns),
                              "apply_start": index[position + 1]})
    schedule = pd.DataFrame(schedule_rows).set_index("rebalance_date")
    assert (schedule["window_end"] < schedule["apply_start"]).all()

    # simulate each strategy: weights drift between rebalances
    gross = {}
    traded = {}
    holdings = {}
    for strategy in strategies:
        name = strategy["name"]
        held = np.zeros(len(tickers))  # all cash before the initial purchase
        daily_parts = []
        traded_list = []
        holdings_list = []
        for k in range(len(positions)):
            if strategy["kind"] == "buyhold" and k > 0:
                target = held  # no trading after the initial purchase
            else:
                target = targets[name][k]
            traded_list.append(np.abs(target - held).sum())
            holdings_list.append(target)
            period = values[positions[k] + 1:last_day[k] + 1]
            growth = np.cumprod(1.0 + period, axis=0)
            value = growth @ target
            previous = np.concatenate(([1.0], value[:-1]))
            daily_parts.append(value / previous - 1.0)
            held = target * growth[-1] / value[-1]
        gross[name] = np.concatenate(daily_parts)
        traded[name] = traded_list
        holdings[name] = pd.DataFrame(holdings_list, index=index[positions], columns=tickers)

    eval_index = index[positions[0] + 1:]
    weights = pd.concat(holdings, names=["strategy", "rebalance_date"]).swaplevel().sort_index(kind="stable")
    return {"gross": pd.DataFrame(gross, index=eval_index), "rf": rf.loc[eval_index], "weights": weights,
            "traded": pd.DataFrame(traded, index=index[positions]), "schedule": schedule, "cost_bps": cost_bps}


def sharpe_ratio(net, rf):
    # mean daily excess return over RF * 252 / (std * sqrt(252))
    excess = net.sub(rf.reindex(net.index), axis=0)
    return excess.mean() * config.TRADING_DAYS / (excess.std() * np.sqrt(config.TRADING_DAYS))


def performance_table(result, rf):
    # net CAGR, volatility, Sharpe, drawdown, turnover, cost drag, daily VaR/CVaR and worst full year
    net = net_returns(result, result["cost_bps"])
    gross = result["gross"]
    years = len(net) / config.TRADING_DAYS
    cagr = (1 + net).prod() ** (1 / years) - 1
    gross_cagr = (1 + gross).prod() ** (1 / years) - 1

    nav = (1 + net).cumprod()
    max_drawdown = (nav / np.maximum(nav.cummax(), 1.0) - 1).min()

    quantile = net.quantile(1 - config.VAR_LEVELS[0])
    cvar = -net.where(net <= quantile).mean()

    # one-way turnover, leaving out the initial purchase
    traded = result["traded"]
    turnover = traded.iloc[1:].mean() * (len(traded) / years) / 2

    # worst return over full calendar years only
    annual = (1 + net).groupby(net.index.year).prod() - 1
    full_years = []
    for year in annual.index:
        days = rf.index[rf.index.year == year]
        if days[0].month == 1 and days[-1].month == 12 and days[0] >= net.index[0] and days[-1] <= net.index[-1]:
            full_years.append(year)

    return pd.DataFrame({
        "cagr": cagr,
        "ann_vol": net.std() * np.sqrt(config.TRADING_DAYS),
        "sharpe": sharpe_ratio(net, rf),
        "max_drawdown": max_drawdown,
        "annual_turnover": turnover,
        "cost_drag": gross_cagr - cagr,
        "var_95": -quantile,
        "cvar_95": cvar,
        "worst_full_year": annual.loc[full_years].min(),
    })


def cost_sensitivity(result, rf):
    # net Sharpe at each cost level, recomputed from stored gross returns and traded amounts
    sensitivity = {}
    for bps in config.COST_SENSITIVITY_BPS:
        sensitivity[bps] = sharpe_ratio(net_returns(result, bps), rf)
    return pd.DataFrame(sensitivity)
