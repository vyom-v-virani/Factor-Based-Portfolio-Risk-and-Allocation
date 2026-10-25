# All editable settings live here.

# Data
TICKERS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG", "GLD", "XLE", "XLF", "XLK"]
BENCHMARK = "SPY"
START_DATE = "2014-01-01"
END_DATE = None  # None means today
TRAIN_FRAC = 0.70
TRADING_DAYS = 252
CACHE_DIR = "data_cache"
OUTPUT_DIR = "outputs"
FF_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"

# Factor regressions
FACTOR_COLS = ["Mkt-RF", "SMB", "HML"]
HAC_LAGS = 5

# Allocation
MAX_WEIGHT = 0.30  # cap applied to min-variance only
RISK_LEVELS = [1, 2, 3, 4, 5]
RISK_LABELS = {1: "Conservative", 2: "Moderately conservative", 3: "Balanced", 4: "Growth", 5: "Aggressive"}
ASSET_CLASS = {
    "SPY": "Equity", "QQQ": "Equity", "IWM": "Equity", "EFA": "Equity", "EEM": "Equity",
    "XLE": "Equity", "XLF": "Equity", "XLK": "Equity",
    "TLT": "Rates", "IEF": "Rates", "LQD": "Credit", "HYG": "Credit", "GLD": "Gold",
}
ASSET_CLASS_ORDER = ["Equity", "Rates", "Credit", "Gold"]
EQUITY_TICKERS = []
for ticker in ASSET_CLASS:
    if ASSET_CLASS[ticker] == "Equity":
        EQUITY_TICKERS.append(ticker)

# Simulation
SEED = 42
N_PATHS = 10000
HORIZON_DAYS = 252
BLOCK_DAYS = 10
VAR_LEVELS = [0.95, 0.99]
STRESS_MARKET_SHOCKS = [-0.10, -0.20]
SCENARIOS = {
    "COVID crash": ("2020-02-19", "2020-03-23"),
    "2022 full year": ("2022-01-03", "2022-12-30"),
}

# Walk-forward backtest
WINDOW_DAYS = 756
REBALANCE_FREQ = "M"  # month-end (last trading day of each month)
COST_BPS = 5
COST_SENSITIVITY_BPS = [0, 5, 10]
BENCH_6040 = {"SPY": 0.6, "IEF": 0.4}
