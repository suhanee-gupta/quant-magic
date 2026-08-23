import numpy as np
import pandas as pd

from quant_magic import backtest, metrics, optimizer


def test_calculate_metrics_basic():
    returns = pd.Series([0.01, -0.02, 0.03, 0.04, -0.01, 0.02, 0.01] * 3)

    result = metrics.calculate_metrics(returns, "Test portfolio")

    assert result["cagr"] is not None
    assert result["sharpe"] is not None
    assert 0 <= result["win_pct"] <= 1
    assert result["max_drawdown"] >= 0


def test_get_markowitz_weights_valid():
    rng = np.random.default_rng(42)
    returns = pd.DataFrame(
        {
            "A": rng.normal(0.0008, 0.01, 200),
            "B": rng.normal(0.0012, 0.012, 200),
            "C": rng.normal(0.0005, 0.011, 200),
        }
    )

    weights = optimizer.get_markowitz_weights(returns)

    assert np.isclose(weights.sum(), 1.0, atol=1e-6)
    assert np.all(weights >= -1e-9)
    assert np.all(np.isfinite(weights))


def test_build_split_returns_three_periods():
    idx = pd.date_range("2020-01-01", periods=900, freq="B")
    df = pd.DataFrame({"A": np.linspace(1, 2, len(idx))}, index=idx)

    train, val, test = backtest.build_split(df, "2021-12-31", "2022-12-31")

    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0
    assert train.index[-1] < val.index[0]
    assert val.index[-1] < test.index[0]
