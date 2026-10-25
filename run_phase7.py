import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import checks
import config
from backtest import cost_sensitivity, net_returns, performance_table, walk_forward
from data import build_dataset, train_test_split
from models import build_dial_portfolios, ledoit_wolf_cov, portfolio_vol

# the optimizer clips tiny out-of-bounds trial steps at many rebalances; final weights are checked below
warnings.filterwarnings("ignore", message="Values in x were outside bounds")

data = build_dataset()  # full sample: the walk-forward uses every day after the first window
returns = data["returns"]
rf = data["factors"]["RF"]
tickers = list(returns.columns)
bench = config.BENCHMARK

# strategies: five dial levels, equal weight, risk parity, benchmark buy-and-hold and 60/40
strategies = []
for level in config.RISK_LEVELS:
    strategies.append({"name": f"Level {level:g}", "kind": "dial", "level": level})
strategies.append({"name": "Equal weight", "kind": "equal"})
strategies.append({"name": "Risk parity", "kind": "parity"})
strategies.append({"name": bench, "kind": "buyhold", "weights": {bench: 1.0}})
strategies.append({"name": "60/40", "kind": "fixed", "weights": config.BENCH_6040})
level_names = [f"Level {level:g}" for level in config.RISK_LEVELS]

result = walk_forward(returns, rf, tickers, strategies, config.COST_BPS)
net = net_returns(result, config.COST_BPS)
gross = result["gross"]
schedule = result["schedule"]
performance = performance_table(result, rf)
sensitivity = cost_sensitivity(result, rf)
years = len(net) / config.TRADING_DAYS

# in-sample (TRAIN) volatility of each dial level, for comparison with the realized volatility
train, test = train_test_split(data)
train_cov, shrinkage = ledoit_wolf_cov(train["returns"])
dial = build_dial_portfolios(train_cov, config.RISK_LEVELS)
train_vols = []
for level in config.RISK_LEVELS:
    train_vols.append(portfolio_vol(dial[level], train_cov))
vol_table = pd.DataFrame({"train_vol": train_vols, "realized_vol": performance.loc[level_names, "ann_vol"].values},
                         index=level_names)

print(f"\nFull sample: {returns.index.min().date()} -> {returns.index.max().date()} ({len(returns)} days); "
      f"window = {config.WINDOW_DAYS} days, cost = {config.COST_BPS:g} bps, rebalance = month-end")
print(f"First rebalance date: {schedule.index[0].date()}  |  last date: {net.index[-1].date()}  |  "
      f"rebalances: {len(schedule)}  |  evaluation period: {years:.2f} years "
      f"({net.index[0].date()} -> {net.index[-1].date()})")

shown = pd.DataFrame({
    "CAGR": performance["cagr"].map("{:.1%}".format), "vol": performance["ann_vol"].map("{:.1%}".format),
    "Sharpe": performance["sharpe"].map("{:.2f}".format), "maxDD": performance["max_drawdown"].map("{:.1%}".format),
    "turnover/yr": performance["annual_turnover"].map("{:.1%}".format),
    "cost drag": performance["cost_drag"].map("{:.2%}".format),
    "1d VaR95": performance["var_95"].map("{:.2%}".format), "1d CVaR95": performance["cvar_95"].map("{:.2%}".format),
    "worst yr": performance["worst_full_year"].map("{:.1%}".format)})
print("\n=== PERFORMANCE (net of costs) ===")
print(shown.to_string())
print("Turnover is one-way and excludes the initial purchase; worst year uses full calendar "
      "years only; VaR/CVaR are realized daily historical losses.")
print("\n=== COST SENSITIVITY: net Sharpe at each cost (bps per unit traded) ===")
sensitivity_shown = sensitivity.copy()
sensitivity_shown.columns = [f"{bps:g} bps" for bps in sensitivity.columns]
print(sensitivity_shown.round(2).to_string())
print("\n=== REALIZED vs IN-SAMPLE VOLATILITY (dial levels) ===")
print(pd.DataFrame({"train (in-sample)": vol_table["train_vol"].map("{:.1%}".format),
                    "walk-forward realized": vol_table["realized_vol"].map("{:.1%}".format)}).to_string())

# save the tables
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
performance.to_csv(f"{config.OUTPUT_DIR}/performance_phase7.csv", index_label="strategy")
sensitivity_saved = sensitivity.copy()
sensitivity_saved.columns = [f"sharpe_{bps:g}bps" for bps in sensitivity.columns]
sensitivity_saved.to_csv(f"{config.OUTPUT_DIR}/cost_sensitivity_phase7.csv", index_label="strategy")
pd.concat([gross.add_prefix("gross_"), net.add_prefix("net_")], axis=1).to_parquet(
    f"{config.OUTPUT_DIR}/daily_returns_phase7.parquet")
result["weights"].reset_index().to_parquet(f"{config.OUTPUT_DIR}/weights_history_phase7.parquet")

# chart 1: net growth of $1 on a log axis (top) with drawdowns underneath
middle = config.RISK_LEVELS[len(config.RISK_LEVELS) // 2]
chart_names = [f"Level {config.RISK_LEVELS[0]:g}", f"Level {middle:g}", f"Level {config.RISK_LEVELS[-1]:g}",
               "Equal weight", "Risk parity", bench, "60/40"]
nav = (1.0 + net[chart_names]).cumprod()
start_row = pd.DataFrame(1.0, index=[net.index[0] - pd.Timedelta(days=1)], columns=chart_names)
nav = pd.concat([start_row, nav])
drawdown = nav / nav.cummax() - 1.0
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
for i, name in enumerate(chart_names):
    color = plt.get_cmap("tab10")(i)
    ax.plot(nav.index, nav[name], color=color, label=name)
    ax2.plot(drawdown.index, drawdown[name] * 100, color=color, lw=1)
ax.set_yscale("log")
ax.set_ylabel("Growth of $1 (net, log scale)")
ax.set_title("Walk-forward backtest: net cumulative growth")
ax.legend(ncol=2)
ax2.set_ylabel("Drawdown (%)")
fig.tight_layout()
fig.savefig(f"{config.OUTPUT_DIR}/equity_curves_phase7.png", dpi=150)
plt.close(fig)

# chart 2: net Sharpe by strategy
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(performance.index, performance["sharpe"], color=plt.get_cmap("tab10")(0))
ax.set_ylabel("Net Sharpe ratio")
ax.set_title("Walk-forward net Sharpe by strategy")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
fig.tight_layout()
fig.savefig(f"{config.OUTPUT_DIR}/sharpe_by_strategy_phase7.png", dpi=150)
plt.close(fig)

# chart 3: in-sample vs walk-forward realized volatility per dial level
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(vol_table))
ax.bar(x - 0.2, vol_table["train_vol"] * 100, width=0.4, label="Train (in-sample)")
ax.bar(x + 0.2, vol_table["realized_vol"] * 100, width=0.4, label="Walk-forward realized")
ax.set_xticks(x, vol_table.index)
ax.set_ylabel("Annualized volatility (%)")
ax.set_title("Dial volatility: in-sample vs walk-forward")
ax.legend()
fig.tight_layout()
fig.savefig(f"{config.OUTPUT_DIR}/realized_vs_insample_vol_phase7.png", dpi=150)
plt.close(fig)
print(f"\nSaved 7 files to {config.OUTPUT_DIR}/ (2 CSV, 2 parquet, 3 PNG)")

# look-ahead test: rerun on data cut part-way and compare the weights and returns it shares with the full run
middle_date = schedule.index[len(schedule) // 2]
cut = returns.index.get_loc(middle_date) + 5
short = walk_forward(returns.iloc[:cut + 1], rf.iloc[:cut + 1], tickers, strategies, config.COST_BPS)
weights_full = result["weights"].loc[short["weights"].index]
weight_diff = (weights_full - short["weights"]).abs().max().max()
return_diff = (gross.loc[short["gross"].index] - short["gross"]).abs().max().max()

weights = result["weights"]
dial_weights = weights[weights.index.get_level_values("strategy").isin(level_names)]
sum_error = (weights.sum(axis=1) - 1).abs().max()
bench_net_no_cost = net_returns(result, 0.0)[bench]
bench_actual = returns[bench].loc[net.index]
nav_diff = ((1 + bench_net_no_cost).cumprod() - (1 + bench_actual).cumprod()).abs().max()
smallest_gap = (gross - net).min().min()
last_of_month = returns.index.to_series().groupby(returns.index.to_period(config.REBALANCE_FREQ)).max()
expected_rebalances = ((last_of_month >= schedule.index[0]) & (last_of_month < returns.index[-1])).sum()
bench_cost = gross[bench] - net[bench]
bench_cost_ok = abs(bench_cost.iloc[0] - config.COST_BPS / 10000) < 1e-15 and bench_cost.iloc[1:].abs().max() == 0.0

print("\n=== SANITY CHECKS ===")
checks.check("no look-ahead: window end < first applied return date, and truncating future data "
             "leaves earlier weights and returns unchanged",
             (schedule["window_end"] < schedule["apply_start"]).all() and weight_diff < 1e-10 and return_diff < 1e-12,
             f"{len(schedule)} rebalances checked; truncated run: {len(short['schedule'])} rebalances, "
             f"max weight diff {weight_diff:.1e}, max return diff {return_diff:.1e}")
checks.check(f"every target sums to 1 (tol 1e-08) and >= -tol; dial weights <= {config.MAX_WEIGHT} + tol",
             sum_error < 1e-8 and (weights >= -1e-8).all().all() and (dial_weights <= config.MAX_WEIGHT + 1e-8).all().all(),
             f"max |sum-1| = {sum_error:.1e}, min weight = {weights.min().min():.1e}, "
             f"max dial weight = {dial_weights.max().max():.4f}")
checks.check(f"cost 0: {bench} buy-and-hold NAV == actual {bench} cumulative return (max abs diff < 1e-08)",
             nav_diff < 1e-8, f"max abs diff = {nav_diff:.2e}")
checks.check("net returns <= gross returns on every day, every strategy", smallest_gap >= 0.0,
             f"min (gross - net) = {smallest_gap:.2e}")
checks.check(f"first rebalance uses >= {config.WINDOW_DAYS} observations", (schedule["n_obs"] >= config.WINDOW_DAYS).all(),
             f"first = {int(schedule['n_obs'].iloc[0])}, min = {int(schedule['n_obs'].min())}")
checks.check("number of rebalances == month-ends from first rebalance to the day before the last date",
             len(schedule) == expected_rebalances, f"{len(schedule)} vs {expected_rebalances}")
checks.check(f"cost drag >= 0 for every strategy; {bench} cost only on the initial purchase",
             (performance["cost_drag"] >= -1e-12).all() and bench_cost_ok,
             f"min drag = {performance['cost_drag'].min():.2e}, {bench} drag = {performance.loc[bench, 'cost_drag']:.2e}, "
             f"{bench} initial cost = {bench_cost.iloc[0]:.1e}, later = {bench_cost.iloc[1:].abs().max():.1e}")

vols = performance["ann_vol"]
top = level_names[-1]
relative = abs(vols[top] / vols[bench] - 1.0)
print("\n=== EXPECTED-BEHAVIOR CHECKS ===")
checks.check("realized walk-forward volatility strictly rises from level 1 to level 5",
             (np.diff(vols[level_names].values) > 0).all(), " < ".join(f"{value:.1%}" for value in vols[level_names]))
checks.check(f"{top} volatility within 20% of {bench} volatility", relative < 0.20,
             f"{vols[top]:.1%} vs {vols[bench]:.1%} (rel {relative:.1%})")
checks.check(f"{level_names[0]} max drawdown smaller than {bench} max drawdown",
             performance.loc[level_names[0], "max_drawdown"] > performance.loc[bench, "max_drawdown"],
             f"{performance.loc[level_names[0], 'max_drawdown']:.1%} vs {performance.loc[bench, 'max_drawdown']:.1%}")
print(f"\n{checks.passed}/{checks.total} checks passed.")
