import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import checks
import config
from data import build_dataset, train_test_split
from models import (build_dial_portfolios, dial_profile, equal_weight, growth_anchor, ledoit_wolf_cov,
                    min_variance, portfolio_vol, risk_parity, weights_for_level)


def print_profile(profile):
    # weights (non-zero rows), volatility and asset-class weight vs risk for one level
    print(f"\n=== LEVEL {profile['level']:g}: {profile['label']} ===")
    weights = profile["weights"]
    print((weights[weights.abs() > 1e-8] * 100).round(1).rename("weight %").to_string())
    print(f"\nAnnualized volatility (train): {profile['volatility']:.2%}")
    mix = pd.DataFrame({"weight %": (profile["class_mix"] * 100).round(1),
                        "risk %": (profile["class_risk"] * 100).round(1)})
    print(mix.to_string())


parser = argparse.ArgumentParser(description="Risk dial portfolios.")
parser.add_argument("--level", type=float, default=None, help="single level in [1, 5]")
args = parser.parse_args()

data = build_dataset()
train, test = train_test_split(data)
cov, shrinkage = ledoit_wolf_cov(train["returns"])
tickers = list(cov.index)

if args.level is not None:
    weights = build_dial_portfolios(cov, [args.level])[args.level]
    print_profile(dial_profile(args.level, cov, weights))
    raise SystemExit

levels = config.RISK_LEVELS
portfolios = build_dial_portfolios(cov, levels)
profiles = []
for level in levels:
    profiles.append(dial_profile(level, cov, portfolios[level]))
print(f"\nTRAIN: {train['returns'].index.min().date()} -> {train['returns'].index.max().date()} "
      f"({len(train['returns'])} days)")
for profile in profiles:
    print_profile(profile)

# summary table: one row per level
class_columns = []
for asset_class in config.ASSET_CLASS_ORDER:
    class_columns.append(asset_class.lower() + "_weight")
rows = []
for profile in profiles:
    row = {"level": profile["level"], "label": profile["label"], "ann_vol": profile["volatility"]}
    for asset_class, column in zip(config.ASSET_CLASS_ORDER, class_columns):
        row[column] = profile["class_mix"][asset_class]
    row["largest_position"] = f"{profile['weights'].idxmax()} {profile['weights'].max():.1%}"
    row["largest_position_weight"] = profile["weights"].max()
    rows.append(row)
summary = pd.DataFrame(rows).set_index("level")
shown = summary.drop(columns="largest_position_weight")
for column in ["ann_vol"] + class_columns:
    shown[column] = shown[column].map("{:.1%}".format)
print("\n=== SUMMARY ===")
print(shown.to_string())

w_equal = equal_weight(tickers)
w_parity, parity_ok = risk_parity(cov)
references = {"Equal weight (all)": portfolio_vol(w_equal, cov), "Risk parity": portfolio_vol(w_parity, cov),
              config.BENCHMARK: train["returns"][config.BENCHMARK].std() * np.sqrt(config.TRADING_DAYS)}
print("\n=== REFERENCE VOLATILITY (train) ===")
for name in references:
    print(f"{name:<20s} {references[name]:.2%}")

# save the weights and summary, then the two charts
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
weights_table = pd.DataFrame(portfolios)
weights_table.to_csv(f"{config.OUTPUT_DIR}/risk_dial_weights_phase5.csv", index_label="ticker")
summary.to_csv(f"{config.OUTPUT_DIR}/risk_dial_summary_phase5.csv")

fig, ax = plt.subplots(figsize=(8, 5))
bottom = np.zeros(len(profiles))
for i, asset_class in enumerate(config.ASSET_CLASS_ORDER):
    values = np.array([profile["class_mix"][asset_class] for profile in profiles]) * 100
    ax.bar([f"{p['level']:g}\n{p['label']}" for p in profiles], values, bottom=bottom,
           color=plt.get_cmap("tab10")(i), label=asset_class)
    bottom += values
ax.set_ylabel("Weight (%)")
ax.set_title("Asset-class mix by risk level (train)")
ax.legend(title="Asset class", bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig.savefig(f"{config.OUTPUT_DIR}/risk_dial_mix_phase5.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(levels, [p["volatility"] * 100 for p in profiles], marker="o", color="black", label="Risk dial")
for i, name in enumerate(references):
    ax.axhline(references[name] * 100, ls="--", color=plt.get_cmap("tab10")(i), label=f"{name} ({references[name]:.1%})")
ax.set_xticks(levels)
ax.set_xlabel("Risk level")
ax.set_ylabel("Annualized volatility (%)")
ax.set_title("Realized volatility by risk level (train)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{config.OUTPUT_DIR}/risk_dial_vol_phase5.png", dpi=150)
plt.close(fig)
print(f"\nSaved 4 files to {config.OUTPUT_DIR}/ (2 CSV, 2 PNG)")

# checks
w_minvar, minvar_ok = min_variance(cov, config.MAX_WEIGHT)
growth = growth_anchor(cov.index)
sum_error = (weights_table.sum() - 1).abs().max()
diff_low = (portfolios[min(levels)] - w_minvar).abs().max()
diff_high = (portfolios[max(levels)] - growth).abs().max()
equity_weights = weights_table.loc[config.EQUITY_TICKERS].sum()
vols = pd.Series([p["volatility"] for p in profiles], index=levels)
try:
    weights_for_level(min(levels) - 1, w_minvar, growth)
    raises_low = False
except ValueError:
    raises_low = True
try:
    weights_for_level(max(levels) + 1, w_minvar, growth)
    raises_high = False
except ValueError:
    raises_high = True

print("\n=== SANITY CHECKS ===")
checks.check("every level sums to 1 (tol 1e-08) and all weights >= -tol",
             sum_error < 1e-8 and (weights_table >= -1e-8).all().all(),
             f"max |sum-1| = {sum_error:.2e}, min weight = {weights_table.min().min():.2e}")
checks.check(f"every weight <= {config.MAX_WEIGHT} + tol", (weights_table <= config.MAX_WEIGHT + 1e-8).all().all(),
             f"max weight = {weights_table.max().max():.4f}")
checks.check("level 1 weights equal min-variance weights (max abs diff < 1e-8)", diff_low < 1e-8,
             f"max abs diff = {diff_low:.2e}")
checks.check("level 5 weights equal growth anchor (max abs diff < 1e-8)", diff_high < 1e-8,
             f"max abs diff = {diff_high:.2e}")
checks.check("equity weight strictly increasing across levels", (np.diff(equity_weights.values) > 0).all(),
             " < ".join(f"{value:.1%}" for value in equity_weights.values))
checks.check("annualized volatility strictly increasing across levels", (np.diff(vols.values) > 0).all(),
             " < ".join(f"{value:.2%}" for value in vols.values))
checks.check("weights_for_level raises ValueError for level 0 and 6", raises_low and raises_high,
             f"level 0: {raises_low}, level 6: {raises_high}")
print("\n=== EXPECTED-BEHAVIOR CHECKS ===")
checks.check("level 5 annualized vol between 15% and 22%", 0.15 <= vols.iloc[-1] <= 0.22, f"{vols.iloc[-1]:.2%}")
print(f"\n{checks.passed}/{checks.total} checks passed.")
