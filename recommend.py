import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import checks
import config
from data import build_dataset, load_ff_factors, load_prices, train_test_split
from models import (asset_class_summary, build_dial_portfolios, dial_profile, fit_factor_models,
                    ledoit_wolf_cov, weights_for_level)
from simulate import factor_exposures, reconstruct_returns, risk_stats, simulate_portfolios


def load_price_returns(refresh=False):
    # daily simple returns from all available prices (not cut back to the factor dates)
    prices = load_prices(refresh)
    return prices.pct_change(fill_method=None).iloc[1:].dropna()


def trailing_window(returns, asof):
    # the last WINDOW_DAYS observations on or before the as-of date
    available = returns.loc[:asof]
    if len(available) < config.WINDOW_DAYS:
        raise ValueError(f"Only {len(available)} observations on or before {asof}; need {config.WINDOW_DAYS}.")
    return available.iloc[-config.WINDOW_DAYS:]


def allocation_as_of(returns, asof, level):
    # dial weights, covariance and window computed from data on or before the as-of date
    window = trailing_window(returns, asof)
    cov, shrinkage = ledoit_wolf_cov(window)
    weights = build_dial_portfolios(cov, [level])[level]
    return weights, cov, window


def previous_month_end(returns, asof_date):
    # last trading day of the month before the as-of date's month (None if there is no earlier month)
    earlier = returns.index[returns.index.to_period("M") < asof_date.to_period("M")]
    if len(earlier) == 0:
        return None
    return earlier.max()


def asset_class_weights(weights):
    # ETF weights summed within each asset class, in the configured order
    split = {}
    for asset_class in config.ASSET_CLASS_ORDER:
        split[asset_class] = 0.0
    for ticker in weights.index:
        split[config.ASSET_CLASS[ticker]] += weights[ticker]
    return pd.Series(split)


def bootstrap_and_exposures(window, factors, weights):
    # factor regressions and block bootstrap on the window days that have Ken French data
    dates = window.index.intersection(factors.index)
    if len(dates) < 100:
        return None
    window_factors = factors.loc[dates]
    excess = window.loc[dates].sub(window_factors["RF"], axis=0)
    models = fit_factor_models(excess, window_factors)
    reconstructed = reconstruct_returns(models, window_factors)
    weights_df = pd.DataFrame({"allocation": weights})
    sim = simulate_portfolios(reconstructed, window_factors["RF"], weights_df, config.SEED)
    stats = risk_stats(sim["horizon"])
    exposures = factor_exposures(weights_df, models["params"])
    return {"var_95": stats.loc["allocation", "var_95"], "cvar_95": stats.loc["allocation", "cvar_95"],
            "betas": exposures.loc["allocation"], "n_days": len(dates), "first_date": dates.min(),
            "last_date": dates.max()}


def loss_text(value, no_loss_text):
    # a loss as a percent, or a message when the value is a gain
    if value < 0:
        return no_loss_text
    return f"{value:.1%}"


def print_risk_block(title, note, risk):
    # one block of 1-year tail risk: VaR and CVaR as losses and the 5th-percentile return
    if risk is None:
        print(f"{title}: not enough days with Ken French data")
        return
    print(f"{title} ({risk['first_date'].date()} to {risk['last_date'].date()}, {risk['n_days']:,} days)  [{note}]")
    print(f"    95% VaR: {loss_text(risk['var_95'], 'no loss at the 5th percentile')}   "
          f"95% CVaR: {loss_text(risk['cvar_95'], 'no loss on average in the worst 5%')}   "
          f"5th-percentile 1-year return: {-risk['var_95']:+.1%}")


def history_asset_class_split(history, level):
    # asset-class split over time from the stored Phase 7 target weights (blended for fractional levels)
    low = history[history["strategy"] == f"Level {min(config.RISK_LEVELS)}"].set_index("rebalance_date")[config.TICKERS]
    high = history[history["strategy"] == f"Level {max(config.RISK_LEVELS)}"].set_index("rebalance_date")[config.TICKERS]
    if level in config.RISK_LEVELS:
        table = history[history["strategy"] == f"Level {level:g}"].set_index("rebalance_date")[config.TICKERS]
    else:
        table = weights_for_level(level, low, high)
    split = pd.DataFrame(index=table.index)
    for asset_class in config.ASSET_CLASS_ORDER:
        members = []
        for ticker in config.TICKERS:
            if config.ASSET_CLASS[ticker] == asset_class:
                members.append(ticker)
        split[asset_class] = table[members].sum(axis=1)
    return split


def plot_history(split, level, path):
    # stacked area chart of the asset-class split over the walk-forward rebalance dates
    fig, ax = plt.subplots(figsize=(10, 5))
    series = []
    colors = []
    for i, asset_class in enumerate(config.ASSET_CLASS_ORDER):
        series.append(split[asset_class] * 100)
        colors.append(plt.get_cmap("tab10")(i))
    ax.stackplot(split.index, series, labels=config.ASSET_CLASS_ORDER, colors=colors)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Weight (%)")
    ax.set_title(f"Asset-class split over time, level {level:g} (Phase 7 walk-forward target weights)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), title="Asset class")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(level, asof_text, refresh):
    returns = load_price_returns(refresh)
    factors = load_ff_factors(refresh)
    if refresh:
        print(f"Downloaded fresh prices and factors; overwrote {config.SNAPSHOT_DIR}/")

    # the as-of date is the latest trading day on or before the requested date
    if asof_text is None:
        requested = returns.index.max()
    else:
        requested = pd.Timestamp(asof_text)
    available = returns.index[returns.index <= requested]
    if len(available) == 0:
        raise ValueError(f"No price data on or before {requested.date()}.")
    asof_date = available.max()

    weights, cov, window = allocation_as_of(returns, asof_date, level)
    profile = dial_profile(level, cov, weights)
    class_split = profile["class_mix"]

    # the same allocation as of the previous month-end, for the change columns
    previous_date = previous_month_end(returns, asof_date)
    previous_weights = None
    if previous_date is not None and (returns.index <= previous_date).sum() >= config.WINDOW_DAYS:
        previous_weights, previous_cov, previous_window = allocation_as_of(returns, previous_date, level)
        previous_class_split = asset_class_summary(previous_weights, previous_cov)["weight"]

    # recent: the trailing window; full history: every aligned day from the start through the as-of date
    risk = bootstrap_and_exposures(window, factors, weights)
    risk_full = bootstrap_and_exposures(returns.loc[:asof_date], factors, weights)

    # ---- print
    print(f"\n=== CURRENT ALLOCATION: LEVEL {level:g} ({profile['label']}) ===")
    print(f"As of:            {asof_date.date()}")
    print(f"Window:           {window.index[0].date()} -> {window.index[-1].date()} ({len(window)} trading days)")
    if previous_weights is not None:
        print(f"Last month-end:   {previous_date.date()} (used for the change columns)")
    else:
        print("Last month-end:   not available (not enough earlier history); no change columns")

    split_table = pd.DataFrame({"weight %": (class_split * 100).round(1)})
    if previous_weights is not None:
        split_table["last month %"] = (previous_class_split * 100).round(1)
        split_table["change (pp)"] = ((class_split - previous_class_split) * 100).round(1) + 0.0  # + 0.0 turns -0.0 into 0.0
    print("\n--- Asset-class split ---")
    print(split_table.to_string())

    etf_table = pd.DataFrame({"weight %": (weights * 100).round(1)})
    shown = weights.abs() > 1e-8
    if previous_weights is not None:
        etf_table["last month %"] = (previous_weights * 100).round(1)
        etf_table["change (pp)"] = ((weights - previous_weights) * 100).round(1) + 0.0
        shown = shown | (previous_weights.abs() > 1e-8)
    print("\n--- ETF weights (rows with a non-zero weight now or last month) ---")
    print(etf_table[shown].to_string())

    print(f"\nAnnualized volatility of this allocation over the window: {profile['volatility']:.1%}")
    risk_shares = []
    for asset_class in config.ASSET_CLASS_ORDER:
        risk_shares.append(f"{asset_class} {profile['class_risk'][asset_class]:.1%}")
    print("Share of risk by asset class: " + ", ".join(risk_shares))

    print(f"\n--- 1-year tail risk (block bootstrap: {config.N_PATHS:,} paths x {config.HORIZON_DAYS} days, "
          f"block {config.BLOCK_DAYS}, seed {config.SEED}; weights held fixed) ---")
    print_risk_block("Full history", "primary risk estimate", risk_full)
    print_risk_block("Last 3 years only", "recent regime; understates risk when the window was calm", risk)
    if risk is not None:
        print("\nFactor exposures (Fama-French regressions on the last-3-years sample): " +
              "   ".join(f"{name} {risk['betas'][name]:.2f}" for name in config.FACTOR_COLS))
        print(f"That sample is the {risk['n_days']} of {len(window)} window days that have Ken French data.")
    print(f"Last factor observation: {factors.index.max().date()} (Ken French data lags prices; "
          f"last price date {returns.index.max().date()})")
    print("\nRisk-targeted allocation based on the past 3 years; not a return forecast and not investment advice.")

    # ---- save the table and the history chart
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    rows = []
    for ticker in weights.index:
        row = {"level": level, "as_of": asof_date.date(), "type": "etf", "name": ticker, "weight": weights[ticker],
               "last_month_weight": np.nan, "change_pp": np.nan}
        if previous_weights is not None:
            row["last_month_weight"] = previous_weights[ticker]
            row["change_pp"] = (weights[ticker] - previous_weights[ticker]) * 100
        rows.append(row)
    for asset_class in config.ASSET_CLASS_ORDER:
        row = {"level": level, "as_of": asof_date.date(), "type": "asset_class", "name": asset_class,
               "weight": class_split[asset_class], "last_month_weight": np.nan, "change_pp": np.nan}
        if previous_weights is not None:
            row["last_month_weight"] = previous_class_split[asset_class]
            row["change_pp"] = (class_split[asset_class] - previous_class_split[asset_class]) * 100
        rows.append(row)
    csv_path = f"{config.OUTPUT_DIR}/current_allocation_level{level:g}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    history_path = f"{config.OUTPUT_DIR}/weights_history_phase7.parquet"
    chart_path = f"{config.OUTPUT_DIR}/allocation_history_level{level:g}.png"
    if os.path.exists(history_path):
        history = pd.read_parquet(history_path)
        plot_history(history_asset_class_split(history, level), level, chart_path)
        print(f"\nSaved: {csv_path}, {chart_path}")
    else:
        history = None
        print(f"\nSaved: {csv_path}. History chart skipped: {history_path} not found (run run_phase7.py).")

    # ---- checks
    print("\n=== CHECKS ===")
    checks.check("weights sum to 1, are all >= 0, and none exceeds MAX_WEIGHT",
                 abs(weights.sum() - 1) < 1e-8 and (weights >= -1e-8).all() and (weights <= config.MAX_WEIGHT + 1e-8).all(),
                 f"sum = {weights.sum():.10f}, min = {weights.min():.2e}, max = {weights.max():.4f}")
    checks.check("last return date used is on or before the as-of date, and exactly WINDOW_DAYS observations",
                 window.index.max() <= asof_date and len(window) == config.WINDOW_DAYS,
                 f"last used {window.index.max().date()} <= {asof_date.date()}, {len(window)} observations")
    rollup = asset_class_weights(weights)
    rollup_diff = (rollup - class_split).abs().max()
    checks.check("asset-class split sums to 100% and the ETF weights roll up to it exactly",
                 abs(class_split.sum() - 1) < 1e-8 and rollup_diff < 1e-12,
                 f"split sums to {class_split.sum():.2%}, max roll-up diff {rollup_diff:.1e}")
    if previous_weights is not None:
        change_sum = (weights - previous_weights).sum()
        checks.check("changes vs last month sum to zero across ETFs", abs(change_sum) < 1e-8,
                     f"sum of changes = {change_sum:.2e}")
    else:
        print("[SKIP] changes vs last month: no previous month-end allocation available")
    if history is not None:
        # consistency: the allocation as of the last Phase 7 rebalance date must equal the stored Phase 7 weights
        last_rebalance = history["rebalance_date"].max()
        anchor_cov, anchor_shrinkage = ledoit_wolf_cov(trailing_window(returns, last_rebalance))
        test_levels = [1, 3, 5]
        recomputed = build_dial_portfolios(anchor_cov, test_levels)
        diffs = []
        for test_level in test_levels:
            row = history[(history["strategy"] == f"Level {test_level}") & (history["rebalance_date"] == last_rebalance)]
            stored = row.iloc[0][config.TICKERS].astype(float)
            diffs.append((recomputed[test_level] - stored.reindex(recomputed[test_level].index)).abs().max())
        checks.check(f"consistency: allocation as of the last Phase 7 rebalance ({last_rebalance.date()}) equals the "
                     f"stored Phase 7 weights for levels 1, 3 and 5 (max abs diff < 1e-8)", max(diffs) < 1e-8,
                     "max abs diff by level: " + ", ".join(f"{d:.1e}" for d in diffs))
    else:
        print("[SKIP] consistency check: stored Phase 7 weights not found")
    phase6_path = f"{config.OUTPUT_DIR}/risk_stats_phase6.csv"
    if os.path.exists(phase6_path):
        # the full-history bootstrap as of the end of the train period must reproduce Phase 6 for level 5
        train, test = train_test_split(build_dataset())
        train_end = train["returns"].index.max()
        level5_weights, level5_cov, level5_window = allocation_as_of(returns, train_end, 5)
        level5_risk = bootstrap_and_exposures(returns.loc[:train_end], factors, level5_weights)
        phase6 = pd.read_csv(phase6_path).set_index("portfolio")
        var_diff = abs(level5_risk["var_95"] - phase6.loc["Level 5", "1y_var_95"])
        cvar_diff = abs(level5_risk["cvar_95"] - phase6.loc["Level 5", "1y_cvar_95"])
        checks.check(f"full-history 95% VaR and CVaR for level 5 as of {train_end.date()} (end of the train period) "
                     f"equal the Phase 6 level 5 values (within 1e-6)", max(var_diff, cvar_diff) < 1e-6,
                     f"VaR {level5_risk['var_95']:.8f} vs {phase6.loc['Level 5', '1y_var_95']:.8f}, "
                     f"CVaR {level5_risk['cvar_95']:.8f} vs {phase6.loc['Level 5', '1y_cvar_95']:.8f}")
    else:
        print("[SKIP] Phase 6 consistency check: outputs/risk_stats_phase6.csv not found")
    print(f"\n{checks.passed}/{checks.total} checks passed.")


def main():
    parser = argparse.ArgumentParser(description="Current risk-dial allocation from the trailing window.")
    parser.add_argument("--level", type=float, required=True, help="risk level from 1 to 5 (decimals allowed)")
    parser.add_argument("--asof", default=None, help="compute the allocation as of this date (default: last price date)")
    parser.add_argument("--refresh", action="store_true", help="download fresh prices and factors and overwrite the data snapshot")
    args = parser.parse_args()
    try:
        # same validation and message as the risk dial, checked before anything is downloaded or deleted
        weights_for_level(args.level, pd.Series([0.0]), pd.Series([0.0]))
        run(args.level, args.asof, args.refresh)
    except (ValueError, FileNotFoundError) as error:
        print(f"ERROR: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
