import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import checks
import config
from data import build_dataset, train_test_split
from models import asset_class_summary, contribution_table, equal_weight, ledoit_wolf_cov, min_variance, risk_parity


def format_table(table):
    # percent weights and risks (1 decimal) and the risk-to-weight ratio (2 decimals)
    return pd.DataFrame({"weight %": (table["weight"] * 100).round(1),
                         "risk %": (table["risk_contribution"] * 100).round(1),
                         "risk/weight": table["risk_to_weight_ratio"].round(2)})


data = build_dataset()
train, test = train_test_split(data)
cov, shrinkage = ledoit_wolf_cov(train["returns"])
tickers = list(cov.index)

strategies = {"Equal weight": equal_weight(tickers), "Min variance": min_variance(cov, config.MAX_WEIGHT)[0],
              "Risk parity": risk_parity(cov)[0]}
tables = {}
summaries = {}
for name in strategies:
    tables[name] = contribution_table(strategies[name], cov)
    summaries[name] = asset_class_summary(strategies[name], cov)

print(f"\nTRAIN: {train['returns'].index.min().date()} -> {train['returns'].index.max().date()} "
      f"({len(train['returns'])} days)")
for name in strategies:
    print(f"\n=== {name.upper()}: TICKER CONTRIBUTIONS ===")
    print(format_table(tables[name]).to_string())
    print(f"\n--- {name}: asset-class summary ---")
    print(format_table(summaries[name]).to_string())

print("\n=== HEADLINES ===")
for name in strategies:
    equity = summaries[name].loc["Equity"]
    print(f"{name}: Equity is {equity['weight']:.1%} of capital but {equity['risk_contribution']:.1%} of risk.")

# save the CSV files
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
ticker_parts = []
class_parts = []
for name in strategies:
    ticker_parts.append(tables[name].reset_index(names="ticker").assign(strategy=name))
    class_parts.append(summaries[name].reset_index().assign(strategy=name))
long_table = pd.concat(ticker_parts)[["strategy", "ticker", "weight", "risk_contribution", "risk_to_weight_ratio"]]
long_table.rename(columns={"risk_to_weight_ratio": "ratio"}).to_csv(f"{config.OUTPUT_DIR}/risk_contributions_phase4.csv", index=False)
class_table = pd.concat(class_parts)[["strategy", "asset_class", "weight", "risk_contribution", "risk_to_weight_ratio"]]
class_table.rename(columns={"risk_to_weight_ratio": "ratio"}).to_csv(f"{config.OUTPUT_DIR}/asset_class_summary_phase4.csv", index=False)

# chart 1: weight vs risk contribution per ticker, one panel per strategy, same y-axis
fig, axes = plt.subplots(1, len(tables), figsize=(5.5 * len(tables), 5), sharey=True)
for ax, name in zip(axes, tables):
    table = tables[name].sort_values("weight", ascending=False)
    x = np.arange(len(table))
    ax.bar(x - 0.2, table["weight"] * 100, width=0.4, label="Weight")
    ax.bar(x + 0.2, table["risk_contribution"] * 100, width=0.4, label="Risk contribution")
    ax.set_xticks(x, table.index, rotation=90)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(name)
axes[0].set_ylabel("% of total")
axes[0].legend()
fig.tight_layout()
fig.savefig(f"{config.OUTPUT_DIR}/risk_vs_weight_phase4.png", dpi=150)
plt.close(fig)

# chart 2: stacked bars by asset class, weight on the left and risk contribution on the right
classes = list(summaries["Equal weight"].index)
names = list(summaries)
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
for ax, column, title in zip(axes, ["weight", "risk_contribution"], ["Weight by asset class", "Risk contribution by asset class"]):
    bottom_positive = np.zeros(len(names))
    bottom_negative = np.zeros(len(names))
    for i, asset_class in enumerate(classes):
        values = np.array([summaries[name].loc[asset_class, column] for name in names]) * 100
        bottom = np.where(values >= 0, bottom_positive, bottom_negative)
        ax.bar(names, values, bottom=bottom, color=plt.get_cmap("tab10")(i), label=asset_class)
        bottom_positive += np.where(values >= 0, values, 0)
        bottom_negative += np.where(values < 0, values, 0)
    ax.set_title(title)
    ax.axhline(0, color="black", lw=0.5)
axes[0].set_ylabel("% of total")
axes[1].legend(title="Asset class", bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig.savefig(f"{config.OUTPUT_DIR}/asset_class_risk_phase4.png", dpi=150)
plt.close(fig)
print(f"\nSaved 4 files to {config.OUTPUT_DIR}/ (2 CSV, 2 PNG)")

ticker_sums = []
class_risk_sums = []
class_weight_sums = []
for name in strategies:
    ticker_sums.append(tables[name]["risk_contribution"].sum())
    class_risk_sums.append(summaries[name]["risk_contribution"].sum())
    class_weight_sums.append(summaries[name]["weight"].sum())
missing = []
for ticker in config.TICKERS:
    if ticker not in config.ASSET_CLASS:
        missing.append(ticker)
parity_deviation = (tables["Risk parity"]["risk_contribution"] - 1.0 / len(tickers)).abs().max()
equal_equity = summaries["Equal weight"].loc["Equity"]
minvar_credit = summaries["Min variance"].loc["Credit", "risk_contribution"]


def join_numbers(numbers):
    return ", ".join(f"{number:.6f}" for number in numbers)


print("\n=== SANITY CHECKS ===")
checks.check("ticker risk contributions sum to 1 (every strategy)", np.allclose(ticker_sums, 1.0), join_numbers(ticker_sums))
checks.check("asset-class risk contributions sum to 1", np.allclose(class_risk_sums, 1.0), join_numbers(class_risk_sums))
checks.check("asset-class weights sum to 1", np.allclose(class_weight_sums, 1.0), join_numbers(class_weight_sums))
checks.check("every ticker in TICKERS appears in ASSET_CLASS", len(missing) == 0, f"missing = {missing}")
checks.check("risk parity: all risk contributions within 0.02 of 1/N", parity_deviation < 0.02,
             f"max deviation = {parity_deviation:.2e}")
checks.check("equal weight: Equity share of risk > share of weight", equal_equity["risk_contribution"] > equal_equity["weight"],
             f"risk {equal_equity['risk_contribution']:.1%} vs weight {equal_equity['weight']:.1%}")
print("\n=== EXPECTED-BEHAVIOR CHECKS ===")
checks.check("min variance: Credit share of risk > 40%", minvar_credit > 0.40, f"{minvar_credit:.1%}")
print(f"\n{checks.passed}/{checks.total} checks passed.")
