from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


def fetch_data(tickers, start: str, end: str, *, verbose: bool = True) -> pd.DataFrame:
    """Download adjusted close prices for the provided tickers."""
    tickers = list(dict.fromkeys(tickers))
    if verbose:
        print(f"Downloading data for {len(tickers)} tickers...")

    data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=False)["Adj Close"]

    if isinstance(data, pd.Series):
        data = data.to_frame()

    data = data.dropna(axis=1)
    return data


def save_frame(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
