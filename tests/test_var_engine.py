import numpy as np
import pandas as pd
import pytest

from quant_magic import var_engine


def _synthetic_asset_returns(n_days=600, seed=0):
    rng = np.random.default_rng(seed)
    cov = np.array([
        [1.0e-4, 0.4e-4, 0.2e-4],
        [0.4e-4, 1.5e-4, 0.3e-4],
        [0.2e-4, 0.3e-4, 0.8e-4],
    ])
    idx = pd.date_range("2020-01-01", periods=n_days, freq="B")
    data = rng.multivariate_normal([0.0005, 0.0003, 0.0004], cov, size=n_days)
    return pd.DataFrame(data, index=idx, columns=["A", "B", "C"])


def test_historical_var_matches_empirical_quantile():
    returns = np.linspace(-0.05, 0.05, 101)

    assert var_engine.historical_var(returns, 0.95) == pytest.approx(0.045)
    assert var_engine.historical_var(returns, 0.99) == pytest.approx(0.049)


def test_parametric_var_matches_normal_formula():
    rng = np.random.default_rng(1)
    returns = rng.normal(0.001, 0.02, 500)
    expected = -(returns.mean() - 2.3263478740 * returns.std(ddof=1))

    assert var_engine.parametric_var(returns, 0.99) == pytest.approx(expected, rel=1e-6)


def test_monte_carlo_var_close_to_parametric_for_normal_assets():
    assets = _synthetic_asset_returns()
    weights = pd.Series({"C": 0.2, "A": 0.5, "B": 0.3})
    portfolio = assets @ weights.reindex(assets.columns)

    mc = var_engine.monte_carlo_var(assets, weights, 0.99, n_sims=200_000, rng=np.random.default_rng(3))
    param = var_engine.parametric_var(portfolio, 0.99)

    assert mc == pytest.approx(param, rel=0.03)


def test_kupiec_zero_lr_when_actual_equals_expected():
    hits = np.zeros(200, dtype=bool)
    hits[:10] = True

    result = var_engine.kupiec_pof_test(hits, 0.95)

    assert result["expected"] == pytest.approx(10)
    assert result["actual"] == 10
    assert result["lr"] == pytest.approx(0.0, abs=1e-10)
    assert result["p_value"] == pytest.approx(1.0)


def test_kupiec_no_exceedances_known_value():
    result = var_engine.kupiec_pof_test(np.zeros(250, dtype=bool), 0.99)

    assert result["lr"] == pytest.approx(-2 * 250 * np.log(0.99))
    assert 0.02 < result["p_value"] < 0.03


def test_kupiec_rejects_too_many_exceedances():
    hits = np.zeros(250, dtype=bool)
    hits[::10] = True  # 25 breaches vs 2.5 expected at 99%

    assert var_engine.kupiec_pof_test(hits, 0.99)["p_value"] < 0.001


def test_christoffersen_detects_clustering():
    clustered = np.zeros(250, dtype=bool)
    clustered[100:106] = True
    spread = np.zeros(250, dtype=bool)
    spread[::40] = True

    clustered_result = var_engine.christoffersen_independence_test(clustered)
    spread_result = var_engine.christoffersen_independence_test(spread)

    assert clustered_result["n11"] == 5
    assert clustered_result["p_value"] < 0.01
    assert spread_result["n11"] == 0
    assert spread_result["p_value"] > 0.05


def test_christoffersen_no_exceedances_passes():
    result = var_engine.christoffersen_independence_test(np.zeros(100, dtype=bool))

    assert result["lr"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(1.0)


def test_rolling_forecasts_are_out_of_sample():
    assets = _synthetic_asset_returns()
    weights = np.array([0.5, 0.3, 0.2])
    test_index = assets.index[300:]

    forecasts = var_engine.rolling_var_forecasts(
        assets, weights, test_index, window=250, n_sims=2_000, seed=0
    )

    first = assets.index[300]
    window_portfolio = assets.iloc[50:300] @ weights
    assert forecasts.loc[first, "var_historical_99"] == pytest.approx(
        var_engine.historical_var(window_portfolio, 0.99)
    )
    assert forecasts.loc[first, "portfolio_return"] == pytest.approx(assets.iloc[300] @ weights)
    for method in var_engine.METHODS:
        assert (forecasts[f"var_{method}_99"] > forecasts[f"var_{method}_95"]).all()


def test_rolling_forecasts_require_enough_history():
    assets = _synthetic_asset_returns(n_days=300)

    with pytest.raises(ValueError, match="history"):
        var_engine.rolling_var_forecasts(assets, [1 / 3] * 3, assets.index[100:], window=250)


def test_run_var_analysis_writes_outputs(tmp_path):
    assets = _synthetic_asset_returns()
    weights = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})

    result = var_engine.run_var_analysis(
        assets, weights, assets.index[300],
        n_sims=1_000,
        results_dir=tmp_path / "results",
        plots_dir=tmp_path / "plots",
        verbose=False,
    )

    report = result["report"]
    assert len(report) == len(var_engine.METHODS) * 2
    assert (report["n_obs"] == 300).all()
    assert (tmp_path / "results" / "var_forecasts.csv").exists()
    assert (tmp_path / "results" / "var_backtest.csv").exists()
    assert (tmp_path / "plots" / "var_exceedances_95.png").exists()
    assert (tmp_path / "plots" / "var_exceedances_99.png").exists()
    # Synthetic returns are i.i.d. normal, so the 95% parametric model should be well calibrated.
    row = report[(report["method"] == "parametric") & (report["confidence"] == 0.95)].iloc[0]
    assert row["kupiec_pass"]
