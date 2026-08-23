from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_metrics(returns_series: pd.Series, name: str) -> dict:
    """Calculate standard portfolio KPIs and return them as a dict."""
    returns_series = pd.Series(returns_series).dropna()
    if returns_series.empty:
        raise ValueError(f"No returns available for {name}.")

    cum_returns = (1 + returns_series).cumprod()
    total_return = cum_returns.iloc[-1] - 1
    years = len(returns_series) / 252.0
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

    std = returns_series.std(ddof=1)
    sharpe = np.sqrt(252) * returns_series.mean() / std if std != 0 else np.nan

    rolling_max = cum_returns.cummax()
    drawdown = cum_returns / rolling_max - 1
    max_drawdown = abs(drawdown.min()) if not drawdown.empty else 0.0

    calmar = cagr / max_drawdown if max_drawdown != 0 else np.nan
    win_pct = (returns_series > 0).mean()

    metrics_dict = {
        "name": name,
        "cagr": float(cagr),
        "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "max_drawdown": float(max_drawdown),
        "calmar": float(calmar) if np.isfinite(calmar) else np.nan,
        "win_pct": float(win_pct),
    }

    print(f"--- {name} ---")
    print(f"CAGR:         {metrics_dict['cagr'] * 100:.2f}%")
    print(f"Sharpe Ratio: {metrics_dict['sharpe']:.2f}")
    print(f"Max Drawdown: {metrics_dict['max_drawdown'] * 100:.2f}%")
    print(f"Calmar Ratio: {metrics_dict['calmar']:.2f}")
    print(f"Win%:         {metrics_dict['win_pct'] * 100:.2f}%\n")

    return metrics_dict
