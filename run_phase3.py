import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import checks
import config
from data import build_dataset, train_test_split
from models import equal_weight, ledoit_wolf_cov, min_variance, portfolio_vol, risk_contributions, risk_parity

data = build_dataset()
train, test = train_test_split(data)
cov, shrinkage = ledoit_wolf_cov(train["returns"])
tickers = list(cov.index)

w_equal = equal_weight(tickers)
w_minvar, minvar_ok = min_variance(cov, config.MAX_WEIGHT)
w_parity, parity_ok = risk_parity(cov)
weights = pd.DataFrame({"Equal weight": w_equal, "Min variance": w_minvar, "Risk parity": w_parity})

vols = pd.Series(dtype=float)
risk_table = pd.DataFrame()
for name in weights.columns:
    vols[name] = portfolio_vol(weights[name], cov)
    risk_table[name] = risk_contributions(weights[name], cov)

print(f"\nTRAIN: {train['returns'].index.min().date()} -> {train['returns'].index.max().date()} "
      f"({len(train['returns'])} days)")
print(f"Ledoit-Wolf shrinkage coefficient: {shrinkage:.4f}")
print("\n=== WEIGHTS (%) ===")
print((weights * 100).round(1).to_string())
print("\n=== ANNUALIZED VOLATILITY ===")
print(vols.map("{:.2%}".format).to_string())
print("\n=== FRACTIONAL RISK CONTRIBUTIONS (%) ===")
print((risk_table * 100).round(1).to_string())

os.makedirs(config.OUTPUT_DIR, exist_ok=True)
weights.to_csv(f"{config.OUTPUT_DIR}/weights_phase3.csv", index_label="ticker")
ax = (weights * 100).plot.bar(figsize=(10, 4.5), width=0.8)
ax.set_ylabel("Weight (%)")
ax.set_title("Portfolio weights by strategy (train)")
ax.axhline(0, color="black", lw=0.5)
ax.legend(title=None)
plt.setp(ax.get_xticklabels(), rotation=0)
ax.figure.tight_layout()
ax.figure.savefig(f"{config.OUTPUT_DIR}/weights_phase3.png", dpi=150)
plt.close(ax.figure)
print(f"\nSaved: {config.OUTPUT_DIR}/weights_phase3.csv, {config.OUTPUT_DIR}/weights_phase3.png")

sum_error = (weights.sum() - 1).abs().max()
min_eigenvalue = np.linalg.eigvalsh(cov.values).min()
asymmetry = np.abs(cov.values - cov.values.T).max()
parity_spread = risk_table["Risk parity"].max() - risk_table["Risk parity"].min()
bond_weight = w_minvar[["TLT", "IEF", "LQD"]].sum()
top_equal_risk = risk_table["Equal weight"].idxmax()

print("\n=== SANITY CHECKS ===")
checks.check("optimizer success (min-variance, risk parity)", minvar_ok and parity_ok,
             f"min_var={minvar_ok}, risk_parity={parity_ok}")
checks.check("weights sum to 1 (tol 1e-08)", sum_error < 1e-8, f"max |sum-1| = {sum_error:.2e}")
checks.check("all weights >= -1e-08", (weights >= -1e-8).all().all(), f"min weight = {weights.min().min():.2e}")
checks.check(f"min-variance weights <= {config.MAX_WEIGHT} + tol", (w_minvar <= config.MAX_WEIGHT + 1e-8).all(),
             f"max weight = {w_minvar.max():.4f}")
checks.check("covariance symmetric and min eigenvalue > 0", asymmetry < 1e-12 and min_eigenvalue > 0,
             f"asym = {asymmetry:.1e}, min eig = {min_eigenvalue:.3e}")
sums = risk_table.sum()
checks.check("risk contributions sum to 1 (all strategies)", np.allclose(sums, 1.0),
             "sums = " + ", ".join(f"{value:.6f}" for value in sums))
checks.check("min-variance vol <= equal-weight vol and <= risk-parity vol",
             vols["Min variance"] <= vols["Equal weight"] and vols["Min variance"] <= vols["Risk parity"],
             f"MV {vols['Min variance']:.4%}, EW {vols['Equal weight']:.4%}, RP {vols['Risk parity']:.4%}")
checks.check("risk parity max-min risk contribution < 0.02", parity_spread < 0.02, f"spread = {parity_spread:.5f}")

print("\n=== EXPECTED-BEHAVIOR CHECKS ===")
checks.check("TLT+IEF+LQD combined min-variance weight > 40%", bond_weight > 0.40, f"{bond_weight:.1%}")
checks.check("risk parity: IEF weight > XLE weight", w_parity["IEF"] > w_parity["XLE"],
             f"IEF {w_parity['IEF']:.1%} vs XLE {w_parity['XLE']:.1%}")
checks.check("equal weight's largest risk contribution is an equity ETF",
             top_equal_risk in config.EQUITY_TICKERS, f"{top_equal_risk} at {risk_table['Equal weight'].max():.1%}")
print(f"\n{checks.passed}/{checks.total} checks passed.")
