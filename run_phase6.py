import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import checks
import config
from data import build_dataset, train_test_split
from models import build_dial_portfolios, equal_weight, fit_factor_models, ledoit_wolf_cov, risk_parity
from simulate import (block_bootstrap_indices, factor_exposures, market_stress, reconstruct_returns,
                      risk_stats, scenario_replay, simulate_portfolios)


def var_cvar_ordered(stats):
    # VaR99 >= VaR95, CVaR >= VaR at each level, for every portfolio
    ok_95 = stats["cvar_95"] >= stats["var_95"]
    ok_99 = stats["cvar_99"] >= stats["var_99"]
    ok_levels = stats["var_99"] >= stats["var_95"]
    return (ok_95 & ok_99 & ok_levels).all()


data = build_dataset()
train, test = train_test_split(data)
models = fit_factor_models(train["excess_returns"], train["factors"])
cov, shrinkage = ledoit_wolf_cov(train["returns"])
tickers = list(cov.index)
bench = config.BENCHMARK

# portfolios: five dial levels, equal weight, risk parity and the benchmark (100% SPY)
dial = build_dial_portfolios(cov, config.RISK_LEVELS)
portfolios = {}
for level in config.RISK_LEVELS:
    portfolios[f"Level {level:g}"] = dial[level]
portfolios["Equal weight"] = equal_weight(tickers)
portfolios["Risk parity"] = risk_parity(cov)[0]
portfolios[bench] = pd.Series({ticker: float(ticker == bench) for ticker in tickers})
weights_df = pd.DataFrame(portfolios)
level_names = [f"Level {level:g}" for level in config.RISK_LEVELS]

# simulate from the factor model with alpha left out (the second run checks reproducibility)
reconstructed = reconstruct_returns(models, train["factors"])
sim = simulate_portfolios(reconstructed, train["factors"]["RF"], weights_df, config.SEED)
sim_again = simulate_portfolios(reconstructed, train["factors"]["RF"], weights_df, config.SEED)
year_stats = risk_stats(sim["horizon"])
day_stats = risk_stats(sim["one_day"])
exposures = factor_exposures(weights_df, models["params"])
stress = market_stress(exposures)

print(f"\nTRAIN: {train['returns'].index.min().date()} -> {train['returns'].index.max().date()} "
      f"({len(train['returns'])} days)")
print(f"Simulation: {config.N_PATHS} paths x {config.HORIZON_DAYS} days, circular block bootstrap "
      f"(block={config.BLOCK_DAYS}), seed={config.SEED}, alpha excluded, daily rebalanced")

year_table = pd.DataFrame({
    "mean": year_stats["mean"].map("{:.1%}".format), "median": year_stats["median"].map("{:.1%}".format),
    "VaR95": year_stats["var_95"].map("{:.1%}".format), "CVaR95": year_stats["cvar_95"].map("{:.1%}".format),
    "VaR99": year_stats["var_99"].map("{:.1%}".format), "CVaR99": year_stats["cvar_99"].map("{:.1%}".format),
    "P(loss)": year_stats["prob_loss"].map("{:.1%}".format), "worst": year_stats["worst"].map("{:.1%}".format)})
print("\n=== 1-YEAR RETURNS (losses shown as positive VaR/CVaR) ===")
print(year_table.to_string())
day_table = pd.DataFrame({"VaR95": day_stats["var_95"].map("{:.2%}".format),
                          "CVaR95": day_stats["cvar_95"].map("{:.2%}".format)})
print("\n=== 1-DAY RISK ===")
print(day_table.to_string())
print("\n=== FACTOR EXPOSURES (portfolio betas) ===")
print(exposures.round(2).to_string())

# historical replay: fixed TRAIN-fit weights (daily rebalanced) on actual total returns (excess + RF)
total_returns = data["excess_returns"][tickers].add(data["factors"]["RF"], axis=0)
daily_portfolio_returns = pd.DataFrame(total_returns.values @ weights_df.values, index=total_returns.index,
                                       columns=weights_df.columns)
replay = scenario_replay(daily_portfolio_returns, config.SCENARIOS)
stress_columns = stress.copy()
stress_columns.columns = [f"linear {shock:+.0%}" for shock in stress.columns]
combined = pd.concat([stress_columns, replay.add_suffix(" (replay)")], axis=1)
print("\n=== MARKET STRESS (linear) vs HISTORICAL REPLAY ===")
print(combined.map("{:.1%}".format).to_string())
print("Note: the linear stress columns are a one-factor approximation (market beta x shock) "
      "that ignores SMB, HML, and rates and credit risk.")
print("Note: replay columns are a historical replay, not a forecast; weights are fit on the "
      "TRAIN sample, and both windows fall inside or overlap it, so they are not out-of-sample.")

# save the tables and charts
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
pd.concat([year_stats.add_prefix("1y_"), day_stats.add_prefix("1d_")], axis=1).to_csv(
    f"{config.OUTPUT_DIR}/risk_stats_phase6.csv", index_label="portfolio")
exposures.to_csv(f"{config.OUTPUT_DIR}/factor_exposures_phase6.csv", index_label="portfolio")
stress_saved = stress.copy()
stress_saved.columns = [f"shock_{shock:+.0%}" for shock in stress.columns]
stress_saved.to_csv(f"{config.OUTPUT_DIR}/market_stress_phase6.csv", index_label="portfolio")
replay.to_csv(f"{config.OUTPUT_DIR}/scenario_replay_phase6.csv", index_label="portfolio")

# chart 1: overlaid 1-year return histograms for levels 1, 3, 5 and the benchmark, with VaR 95 lines
middle = config.RISK_LEVELS[len(config.RISK_LEVELS) // 2]
chart_names = [f"Level {config.RISK_LEVELS[0]:g}", f"Level {middle:g}", f"Level {config.RISK_LEVELS[-1]:g}", bench]
horizon = sim["horizon"]
bins = np.linspace(horizon[chart_names].values.min(), horizon[chart_names].values.max(), 80) * 100
fig, ax = plt.subplots(figsize=(10, 5.5))
for i, name in enumerate(chart_names):
    color = plt.get_cmap("tab10")(i)
    ax.hist(horizon[name] * 100, bins=bins, alpha=0.45, color=color, label=name)
    ax.axvline(-year_stats.loc[name, "var_95"] * 100, color=color, ls="--", lw=1.5)
ax.set_xlabel("1-year return (%)  (dashed = VaR 95)")
ax.set_ylabel("Paths")
ax.set_title("Simulated 1-year return distributions")
ax.legend()
fig.tight_layout()
fig.savefig(f"{config.OUTPUT_DIR}/return_distributions_phase6.png", dpi=150)
plt.close(fig)

# chart 2: 1-year CVaR 95 per portfolio
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(year_stats.index, year_stats["cvar_95"] * 100, color=plt.get_cmap("tab10")(0))
ax.set_ylabel("1-year CVaR 95 (% loss)")
ax.set_title("Expected shortfall at 95% by portfolio")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
fig.tight_layout()
fig.savefig(f"{config.OUTPUT_DIR}/cvar_by_portfolio_phase6.png", dpi=150)
plt.close(fig)
print(f"\nSaved 6 files to {config.OUTPUT_DIR}/ (4 CSV, 2 PNG)")

# checks
alpha = models["params"]["alpha"]
reconstruction_error = ((reconstructed + alpha) - train["excess_returns"]).abs().max().max()
reproducible = np.array_equal(sim["horizon"].values, sim_again["horizon"].values) and \
    np.array_equal(sim["one_day"].values, sim_again["one_day"].values)
year_ordered = var_cvar_ordered(year_stats)
day_ordered = var_cvar_ordered(day_stats)
realized_vol = (train["excess_returns"][tickers].values @ weights_df.values).std(axis=0, ddof=1) * np.sqrt(config.TRADING_DAYS)
simulated_vol = sim["one_day"].values.std(axis=0, ddof=1) * np.sqrt(config.TRADING_DAYS)
relative_diff = np.abs(simulated_vol / realized_vol - 1.0)
positions = block_bootstrap_indices(len(reconstructed), config.N_PATHS, config.HORIZON_DAYS, config.BLOCK_DAYS,
                                    np.random.default_rng(config.SEED))
spy_beta = exposures.loc[bench, config.FACTOR_COLS[0]]
var_95 = year_stats["var_95"]
prob_loss = year_stats["prob_loss"]

print("\n=== SANITY CHECKS ===")
checks.check("reconstruction identity: reconstructed + alpha == actual excess returns",
             reconstruction_error < 1e-10, f"max abs diff = {reconstruction_error:.2e}")
checks.check("reproducibility: same seed gives identical simulation outputs", reproducible, f"identical = {reproducible}")
checks.check("VaR99 >= VaR95 and CVaR >= VaR (1y, every portfolio; 1d too)", year_ordered and day_ordered,
             f"1y all ok = {year_ordered}, 1d all ok = {day_ordered}")
checks.check("simulated annualized 1-day vol within 10% of train realized (every portfolio)", (relative_diff < 0.10).all(),
             f"max rel diff = {relative_diff.max():.2%} ({weights_df.columns[int(relative_diff.argmax())]})")
checks.check("block indices in [0, n_obs-1] with shape (n_paths, horizon)",
             positions.shape == (config.N_PATHS, config.HORIZON_DAYS) and positions.min() >= 0 and positions.max() <= len(reconstructed) - 1,
             f"shape {positions.shape}, min {positions.min()}, max {positions.max()} (n_obs={len(reconstructed)})")
checks.check(f"{bench} market beta between 0.9 and 1.1", 0.9 <= spy_beta <= 1.1, f"{spy_beta:.3f}")

print("\n=== EXPECTED-BEHAVIOR CHECKS ===")
checks.check("1-year VaR95 rises (or ties) from level 1 to level 5", (np.diff(var_95[level_names].values) >= 0).all(),
             " -> ".join(f"{value:.1%}" for value in var_95[level_names]))
checks.check(f"Equal weight 1-year VaR95 < {bench} 1-year VaR95", var_95["Equal weight"] < var_95[bench],
             f"{var_95['Equal weight']:.1%} vs {var_95[bench]:.1%}")

# INFO only (not counted): probability of loss is not monotone in risk when Sharpe ratios differ across asset classes
low_level, high_level = level_names[0], level_names[-1]
if prob_loss[low_level] < prob_loss[high_level]:
    verdict = "level 1 lower"
else:
    verdict = "level 1 not lower"
print(f"\nINFO: probability of loss, level 1 vs level 5: {prob_loss[low_level]:.1%} vs {prob_loss[high_level]:.1%} ({verdict})")
print(f"\n{checks.passed}/{checks.total} checks passed.")
