import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import checks
import config
from data import build_dataset, train_test_split
from models import fit_factor_models

data = build_dataset()
train, test = train_test_split(data)
models = fit_factor_models(train["excess_returns"], train["factors"])
params = models["params"]
tvalues = models["tvalues"]
r2 = models["r2"]
residuals = models["residuals"]

print(f"\nFitted on TRAIN only: {train['returns'].index.min().date()} -> {train['returns'].index.max().date()} "
      f"({len(train['returns'])} days), HAC lags = {config.HAC_LAGS}")

alpha_annualized = (params["alpha"] * config.TRADING_DAYS).rename("alpha_annualized")
summary = pd.concat([alpha_annualized.rename("Ann. alpha"), params[config.FACTOR_COLS].round(2), r2.round(3)], axis=1)
summary["Ann. alpha"] = summary["Ann. alpha"].map("{:.2%}".format)
print("\n=== BETAS ===")
print(summary.to_string())
print("\n=== HAC t-STATS ===")
print(tvalues[config.FACTOR_COLS].round(2).to_string())

# save the results table, residuals and heatmap
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
table = pd.concat([params.add_prefix("coef_"), alpha_annualized, tvalues.add_prefix("t_"), r2], axis=1)
table.index.name = "ticker"
table.to_csv(f"{config.OUTPUT_DIR}/betas_table.csv")
residuals.to_parquet(f"{config.OUTPUT_DIR}/residuals.parquet")

betas = params[config.FACTOR_COLS]
limit = np.abs(betas.values).max()
fig, ax = plt.subplots(figsize=(5, 0.45 * len(betas) + 1.5))
image = ax.imshow(betas.values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
ax.set_xticks(range(betas.shape[1]), betas.columns)
ax.set_yticks(range(betas.shape[0]), betas.index)
for i in range(betas.shape[0]):
    for j in range(betas.shape[1]):
        ax.text(j, i, f"{betas.iat[i, j]:.2f}", ha="center", va="center", fontsize=9)
fig.colorbar(image, ax=ax, label="beta")
ax.set_title("Factor betas (train)")
fig.tight_layout()
fig.savefig(f"{config.OUTPUT_DIR}/betas_heatmap.png", dpi=150)
plt.close(fig)
print(f"\nSaved: {config.OUTPUT_DIR}/betas_table.csv, {config.OUTPUT_DIR}/residuals.parquet, {config.OUTPUT_DIR}/betas_heatmap.png")

print("\n=== SANITY CHECKS (expected behavior) ===")
spy_beta = params.loc["SPY", "Mkt-RF"]
checks.check("SPY Mkt-RF beta > 0.9", spy_beta > 0.9, f"SPY Mkt-RF = {spy_beta:.3f}")
checks.check("SPY Mkt-RF beta < 1.1", spy_beta < 1.1, f"SPY Mkt-RF = {spy_beta:.3f}")
checks.check("SPY R2 > 0.9", r2["SPY"] > 0.9, f"SPY R2 = {r2['SPY']:.3f}")
checks.check("TLT R2 < 0.3", r2["TLT"] < 0.3, f"TLT R2 = {r2['TLT']:.3f}")
checks.check("IWM SMB beta > 0.3", params.loc["IWM", "SMB"] > 0.3, f"IWM SMB = {params.loc['IWM', 'SMB']:.3f}")
checks.check("XLK HML beta < 0", params.loc["XLK", "HML"] < 0, f"XLK HML = {params.loc['XLK', 'HML']:.3f}")
checks.check("XLE HML beta > 0", params.loc["XLE", "HML"] > 0, f"XLE HML = {params.loc['XLE', 'HML']:.3f}")
checks.check("residuals shape == train returns shape", residuals.shape == train["returns"].shape,
             f"{residuals.shape} vs {train['returns'].shape}")
max_mean = residuals.mean().abs().max()
checks.check("residual means ~ 0 (abs < 1e-08)", max_mean < 1e-8, f"max |mean| = {max_mean:.2e}")
print(f"\n{checks.passed}/{checks.total} checks passed.")
