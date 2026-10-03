from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import fetch_data
from .metrics import calculate_metrics
from .optimizer import get_markowitz_weights


def build_split(returns: pd.DataFrame, train_end: str, val_end: str):
    """Split daily returns into train, validation, and test sets."""
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")

    train = returns[:train_end]
    val = returns[train_end:val_end]
    test = returns[val_end:]
    return train, val, test


def run_backtest(
    tickers,
    start_date: str,
    end_date: str,
    train_end: str,
    val_end: str,
    benchmark_ticker: str,
    *,
    output_dir: str | Path | None = None,
):
    """Run a simple long-only Markowitz backtest and return results."""
    prices = fetch_data(tickers, start_date, end_date)
    daily_returns = prices.pct_change().dropna()

    bench_prices = fetch_data([benchmark_ticker], start_date, end_date)
    bench_returns = bench_prices.pct_change().dropna().squeeze()

    train_returns, val_returns, test_returns = build_split(daily_returns, train_end, val_end)
    _, _, bench_test_returns = build_split(pd.DataFrame({benchmark_ticker: bench_returns}), train_end, val_end)

    if train_returns.empty:
        raise ValueError("Training period is empty.")
    if test_returns.empty:
        raise ValueError("Test period is empty.")

    weights = get_markowitz_weights(train_returns)
    weights_series = pd.Series(weights, index=train_returns.columns).sort_values(ascending=False)

    portfolio_test_returns = test_returns.dot(weights)
    aligned_data = pd.DataFrame({
        "Portfolio": portfolio_test_returns,
        "Benchmark": bench_test_returns[benchmark_ticker] if isinstance(bench_test_returns, pd.DataFrame) else bench_test_returns,
    }).dropna()

    metrics = {
        "weights": weights_series,
        "portfolio": calculate_metrics(aligned_data["Portfolio"], "Markowitz Optimized Portfolio"),
        "benchmark": calculate_metrics(aligned_data["Benchmark"], "Benchmark"),
        "asset_returns": daily_returns,
        "test_start": test_returns.index[0],
    }

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"weights": weights_series}).to_csv(out_dir / "weights.csv")
        aligned_data.to_csv(out_dir / "performance.csv")

    return metrics
