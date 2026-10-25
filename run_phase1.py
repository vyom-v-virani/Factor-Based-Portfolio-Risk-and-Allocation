import numpy as np
import pandas as pd

import checks
import config
from data import build_dataset, train_test_split

data = build_dataset()
train, test = train_test_split(data)
returns = data["returns"]
excess_returns = data["excess_returns"]
factors = data["factors"]

print(f"Dropped {data['dropped_nan']} return rows containing NaN (missing prices / ticker not yet listed).")
print(f"Dropped {data['dropped_no_factor']} return rows with no Fama-French date (inner join).")

print("\n=== DATASET ===")
print(f"Tickers loaded ({len(returns.columns)}): {', '.join(returns.columns)}")
print(f"Date range: {returns.index.min().date()} -> {returns.index.max().date()}")
print(f"Trading days: {len(returns)}")
for name, part in [("Train", train["returns"]), ("Test ", test["returns"])]:
    print(f"{name}: {part.index.min().date()} -> {part.index.max().date()} ({len(part)} days)")

print("\n=== SHAPES ===")
print(f"returns:        {returns.shape}")
print(f"excess_returns: {excess_returns.shape}")
print(f"factors:        {factors.shape}")

nan_count = returns.isna().sum().sum() + excess_returns.isna().sum().sum() + factors.isna().sum().sum()
print(f"\nNaN count: {nan_count}")

stats = pd.DataFrame({"Ann. Return": train["returns"].mean() * config.TRADING_DAYS,
                      "Ann. Vol": train["returns"].std() * np.sqrt(config.TRADING_DAYS)})
shown = pd.DataFrame(index=stats.index)
for column in stats.columns:
    shown[column] = stats[column].map("{:.2%}".format)
print("\n=== TRAIN: ANNUALIZED MEAN RETURN & VOLATILITY ===")
print(shown.to_string())

corr = train["returns"].corr()
print("\n=== TRAIN: CORRELATION MATRIX ===")
print(corr.round(2).to_string())

spy_vol = stats.loc["SPY", "Ann. Vol"]
tlt_corr = corr.loc["TLT", "SPY"]
max_factor = factors[config.FACTOR_COLS].abs().max().max()

print("\n=== SANITY CHECKS ===")
checks.check("no NaNs", nan_count == 0, f"{nan_count} NaNs")
checks.check("all return series share the same index",
             returns.index.equals(excess_returns.index) and returns.index.equals(factors.index), "indexes equal")
checks.check("factors in decimal scale (abs max < 0.2)", max_factor < 0.2, f"max abs = {max_factor:.4f}")
checks.check("SPY annualized vol in [10%, 30%]", 0.10 <= spy_vol <= 0.30, f"{spy_vol:.2%}")
checks.check("TLT corr with SPY < 0.5 (train)", tlt_corr < 0.5, f"{tlt_corr:.2f}")
checks.check("train end strictly before test start", train["returns"].index.max() < test["returns"].index.min(),
             f"{train['returns'].index.max().date()} < {test['returns'].index.min().date()}")
print(f"\n{checks.passed}/{checks.total} checks passed.")
