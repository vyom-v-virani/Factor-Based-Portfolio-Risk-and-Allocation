import os

import pandas as pd
import yfinance as yf

import config


def load_prices():
    # adjusted close prices, one column per ticker; read from the cache file if it exists
    name = "prices_" + "-".join(sorted(config.TICKERS)) + "_" + config.START_DATE + "_" + str(config.END_DATE or "today") + ".parquet"
    path = os.path.join(config.CACHE_DIR, name)
    if os.path.exists(path):
        return pd.read_parquet(path)
    raw = yf.download(config.TICKERS, start=config.START_DATE, end=config.END_DATE, auto_adjust=True, progress=False)
    prices = raw["Close"][config.TICKERS]
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.index.name = "Date"
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    prices.to_parquet(path)
    return prices


def load_ff_factors():
    # Ken French daily Mkt-RF, SMB, HML, RF as decimals; read from the cache file if it exists
    path = os.path.join(config.CACHE_DIR, "ff_factors_daily.parquet")
    if os.path.exists(path):
        factors = pd.read_parquet(path)
    else:
        raw = pd.read_csv(config.FF_URL, compression="zip", skiprows=3, index_col=0)
        raw = raw[raw.index.str.len() == 8]  # drops the copyright footer line
        raw.index = pd.to_datetime(raw.index, format="%Y%m%d")
        raw.index.name = "Date"
        raw.columns = raw.columns.str.strip()
        factors = raw[config.FACTOR_COLS + ["RF"]].astype(float) / 100
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        factors.to_parquet(path)
    return factors.loc[config.START_DATE:config.END_DATE]


def build_dataset():
    # returns, excess_returns and factors on the inner join of dates, with no NaNs
    prices = load_prices()
    factors = load_ff_factors()

    # a ticker that starts late shortens the whole dataset
    cutoff = pd.Timestamp(config.START_DATE) + pd.Timedelta(days=30)
    for ticker in prices.columns:
        first_date = prices[ticker].first_valid_index()
        if first_date > cutoff:
            print(f"WARNING: {ticker} history starts {first_date.date()}, more than 30 days after "
                  f"{config.START_DATE}; the inner join will shorten the whole dataset to that start.")

    returns = prices.pct_change(fill_method=None).iloc[1:]
    rows_before = len(returns)
    returns = returns.dropna()
    dropped_nan = rows_before - len(returns)

    dates = returns.index.intersection(factors.index)
    dropped_no_factor = len(returns) - len(dates)
    returns = returns.loc[dates]
    factors = factors.loc[dates]

    excess_returns = returns.sub(factors["RF"], axis=0)
    return {"returns": returns, "excess_returns": excess_returns, "factors": factors,
            "dropped_nan": dropped_nan, "dropped_no_factor": dropped_no_factor}


def train_test_split(dataset):
    # split by date order: the first TRAIN_FRAC of days is train, the rest is test
    cut = int(len(dataset["returns"]) * config.TRAIN_FRAC)
    train = {}
    test = {}
    for name in ["returns", "excess_returns", "factors"]:
        train[name] = dataset[name].iloc[:cut]
        test[name] = dataset[name].iloc[cut:]
    assert train["returns"].index.max() < test["returns"].index.min()
    return train, test
