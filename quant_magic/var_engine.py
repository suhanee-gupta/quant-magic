"""Value-at-Risk engine and VaR backtesting.

VaR is forecast one day ahead for every test-period date using a rolling window
of the asset returns that precede it, so each forecast is strictly out-of-sample.
VaR values are reported as positive loss fractions: a 99% VaR of 0.02 means the
model expects a daily loss worse than 2% on 1% of days. A day is an exceedance
(breach) when the realised portfolio return is below -VaR.

Run standalone with ``python -m quant_magic.var_engine``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy

DEFAULT_CONFIDENCE_LEVELS = (0.95, 0.99)
METHODS = ("historical", "parametric", "monte_carlo")
METHOD_LABELS = {
    "historical": "Historical simulation",
    "parametric": "Parametric (normal)",
    "monte_carlo": "Monte Carlo",
}


def _level_tag(confidence: float) -> str:
    return f"{confidence * 100:g}".replace(".", "_")


def _var_column(method: str, confidence: float) -> str:
    return f"var_{method}_{_level_tag(confidence)}"


def _align_weights(weights, columns: pd.Index) -> np.ndarray:
    if isinstance(weights, pd.Series):
        missing = columns.difference(weights.index)
        if len(missing) > 0:
            raise ValueError(f"Weights missing for assets: {list(missing)}")
        return weights.reindex(columns).to_numpy(dtype=float)

    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(columns),):
        raise ValueError("weights must have one entry per asset column.")
    return weights


# ---------------------------------------------------------------------------
# VaR estimators
# ---------------------------------------------------------------------------

def historical_var(portfolio_returns, confidence: float) -> float:
    """Historical-simulation VaR: the empirical (1 - confidence) return quantile."""
    returns = np.asarray(portfolio_returns, dtype=float)
    return float(-np.quantile(returns, 1 - confidence))


def parametric_var(portfolio_returns, confidence: float) -> float:
    """Variance-covariance VaR assuming normally distributed returns."""
    returns = np.asarray(portfolio_returns, dtype=float)
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    return float(-(mu + stats.norm.ppf(1 - confidence) * sigma))


def simulate_portfolio_returns(
    asset_returns: pd.DataFrame,
    weights,
    n_sims: int = 10_000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate portfolio returns from correlated normal asset returns.

    Asset returns are drawn from a multivariate normal with the sample mean and
    covariance of ``asset_returns``, then combined with ``weights``.
    """
    rng = rng if rng is not None else np.random.default_rng()
    w = _align_weights(weights, asset_returns.columns)
    simulated = rng.multivariate_normal(
        asset_returns.mean().to_numpy(),
        asset_returns.cov().to_numpy(),
        size=n_sims,
    )
    return simulated @ w


def monte_carlo_var(
    asset_returns: pd.DataFrame,
    weights,
    confidence: float,
    n_sims: int = 10_000,
    rng: np.random.Generator | None = None,
) -> float:
    """Monte Carlo VaR from simulated correlated asset returns."""
    simulated = simulate_portfolio_returns(asset_returns, weights, n_sims, rng)
    return historical_var(simulated, confidence)


# ---------------------------------------------------------------------------
# VaR backtests
# ---------------------------------------------------------------------------

def kupiec_pof_test(exceedances, confidence: float) -> dict:
    """Kupiec proportion-of-failures likelihood-ratio test (chi-squared, 1 dof)."""
    hits = np.asarray(exceedances, dtype=bool)
    n_obs = hits.size
    if n_obs == 0:
        raise ValueError("No observations to test.")

    p = 1 - confidence
    x = int(hits.sum())
    phat = x / n_obs

    log_l0 = xlogy(n_obs - x, 1 - p) + xlogy(x, p)
    log_l1 = xlogy(n_obs - x, 1 - phat) + xlogy(x, phat)
    lr = max(-2 * (log_l0 - log_l1), 0.0)

    return {
        "n_obs": n_obs,
        "expected": n_obs * p,
        "actual": x,
        "lr": float(lr),
        "p_value": float(stats.chi2.sf(lr, df=1)),
    }


def christoffersen_independence_test(exceedances) -> dict:
    """Christoffersen Markov independence likelihood-ratio test (chi-squared, 1 dof).

    Tests whether a breach today changes the probability of a breach tomorrow,
    i.e. whether exceedances cluster.
    """
    hits = np.asarray(exceedances, dtype=int)
    if hits.size < 2:
        raise ValueError("Need at least two observations for the independence test.")

    prev, curr = hits[:-1], hits[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))

    pi0 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi1 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    log_l0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    log_l1 = (
        xlogy(n00, 1 - pi0) + xlogy(n01, pi0)
        + xlogy(n10, 1 - pi1) + xlogy(n11, pi1)
    )
    lr = max(-2 * (log_l0 - log_l1), 0.0)

    return {
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "lr": float(lr),
        "p_value": float(stats.chi2.sf(lr, df=1)),
    }


# ---------------------------------------------------------------------------
# Rolling forecasts and reporting
# ---------------------------------------------------------------------------

def rolling_var_forecasts(
    asset_returns: pd.DataFrame,
    weights,
    test_index: pd.Index,
    *,
    window: int = 250,
    confidence_levels=DEFAULT_CONFIDENCE_LEVELS,
    n_sims: int = 10_000,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Forecast one-day VaR for each test date from the preceding ``window`` days.

    Returns a frame indexed by test date with the realised portfolio return,
    one VaR column per method/confidence level, and matching breach flags.
    """
    asset_returns = asset_returns.dropna()
    w = _align_weights(weights, asset_returns.columns)
    portfolio_returns = asset_returns @ w
    rng = np.random.default_rng(seed)

    positions = asset_returns.index.get_indexer(test_index)
    if (positions < 0).any():
        raise ValueError("test_index contains dates not present in asset_returns.")
    if positions.min() < window:
        raise ValueError(
            f"Need {window} days of history before the first test date; "
            f"only {positions.min()} available."
        )

    quantiles = np.array([1 - c for c in confidence_levels])
    rows = []
    for pos in positions:
        hist_assets = asset_returns.iloc[pos - window:pos]
        hist_portfolio = portfolio_returns.iloc[pos - window:pos].to_numpy()
        simulated = simulate_portfolio_returns(hist_assets, w, n_sims, rng)

        hist_q = np.quantile(hist_portfolio, quantiles)
        mc_q = np.quantile(simulated, quantiles)
        row = {"portfolio_return": portfolio_returns.iloc[pos]}
        for i, confidence in enumerate(confidence_levels):
            row[_var_column("historical", confidence)] = -hist_q[i]
            row[_var_column("parametric", confidence)] = parametric_var(hist_portfolio, confidence)
            row[_var_column("monte_carlo", confidence)] = -mc_q[i]
        rows.append(row)

    forecasts = pd.DataFrame(rows, index=asset_returns.index[positions])
    for method in METHODS:
        for confidence in confidence_levels:
            col = _var_column(method, confidence)
            forecasts[f"breach_{method}_{_level_tag(confidence)}"] = (
                forecasts["portfolio_return"] < -forecasts[col]
            )
    return forecasts


def backtest_var(
    forecasts: pd.DataFrame,
    confidence_levels=DEFAULT_CONFIDENCE_LEVELS,
    significance: float = 0.05,
) -> pd.DataFrame:
    """Run Kupiec and Christoffersen tests for every method and confidence level.

    The conditional-coverage test (Kupiec + independence, chi-squared 2 dof) is
    included as a joint check. A test passes when its p-value >= ``significance``.
    """
    rows = []
    for method in METHODS:
        for confidence in confidence_levels:
            hits = forecasts[f"breach_{method}_{_level_tag(confidence)}"].to_numpy()
            pof = kupiec_pof_test(hits, confidence)
            ind = christoffersen_independence_test(hits)
            cc_lr = pof["lr"] + ind["lr"]
            cc_p = float(stats.chi2.sf(cc_lr, df=2))
            rows.append({
                "method": method,
                "confidence": confidence,
                "n_obs": pof["n_obs"],
                "expected_exceedances": pof["expected"],
                "actual_exceedances": pof["actual"],
                "exceedance_rate": pof["actual"] / pof["n_obs"],
                "kupiec_lr": pof["lr"],
                "kupiec_p_value": pof["p_value"],
                "kupiec_pass": pof["p_value"] >= significance,
                "christoffersen_lr": ind["lr"],
                "christoffersen_p_value": ind["p_value"],
                "christoffersen_pass": ind["p_value"] >= significance,
                "cond_coverage_lr": cc_lr,
                "cond_coverage_p_value": cc_p,
                "cond_coverage_pass": cc_p >= significance,
            })
    return pd.DataFrame(rows)


def print_backtest_report(report: pd.DataFrame) -> None:
    def verdict(passed: bool) -> str:
        return "PASS" if passed else "FAIL"

    print("--- VaR Backtest ---")
    header = f"{'Method':<24}{'Conf':>6}{'Exp':>8}{'Act':>6}{'Kupiec p':>12}{'Christ. p':>14}"
    print(header)
    for row in report.itertuples(index=False):
        print(
            f"{METHOD_LABELS[row.method]:<24}"
            f"{row.confidence * 100:>5g}%"
            f"{row.expected_exceedances:>8.1f}"
            f"{row.actual_exceedances:>6d}"
            f"{row.kupiec_p_value:>7.3f} {verdict(row.kupiec_pass)}"
            f"{row.christoffersen_p_value:>9.3f} {verdict(row.christoffersen_pass)}"
        )
    print()


# Palette: neutral ink for P&L, blue for the VaR threshold, status-critical for breaches.
_INK_MUTED = "#8a8984"
_INK_SECONDARY = "#52514e"
_VAR_LINE = "#2a78d6"
_BREACH = "#d03b3b"


def plot_exceedances(forecasts: pd.DataFrame, confidence: float, path: str | Path) -> Path:
    """Plot daily P&L against each method's VaR, highlighting breaches."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    tag = _level_tag(confidence)
    pnl = forecasts["portfolio_return"]
    fig, axes = plt.subplots(len(METHODS), 1, figsize=(12, 9), sharex=True, sharey=True)

    for ax, method in zip(axes, METHODS):
        var = forecasts[_var_column(method, confidence)]
        breaches = forecasts[f"breach_{method}_{tag}"]
        n_breach = int(breaches.sum())
        expected = len(forecasts) * (1 - confidence)

        ax.bar(pnl.index, pnl, width=1.0, color=_INK_MUTED, label="Daily P&L")
        ax.plot(var.index, -var, color=_VAR_LINE, linewidth=2, label=f"−VaR ({confidence * 100:g}%)")
        ax.scatter(
            pnl.index[breaches], pnl[breaches],
            s=36, color=_BREACH, edgecolor="white", linewidth=1, zorder=3,
            label="Breach",
        )
        ax.axhline(0, color=_INK_SECONDARY, linewidth=0.6)
        ax.set_title(
            f"{METHOD_LABELS[method]}: {n_breach} breaches (expected {expected:.1f})",
            loc="left", fontsize=11, color="#0b0b0b",
        )
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.grid(axis="y", color="#e5e4e0", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=_INK_SECONDARY, labelsize=9)

    axes[0].legend(loc="lower left", ncol=3, frameon=False, fontsize=9)
    axes[-1].set_xlabel("Date", color=_INK_SECONDARY)
    fig.supylabel("Daily portfolio return", color=_INK_SECONDARY)
    fig.suptitle(f"{confidence * 100:g}% one-day VaR vs realised P&L", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_var_analysis(
    asset_returns: pd.DataFrame,
    weights,
    test_start,
    *,
    window: int = 250,
    confidence_levels=DEFAULT_CONFIDENCE_LEVELS,
    n_sims: int = 10_000,
    seed: int | None = 42,
    significance: float = 0.05,
    results_dir: str | Path | None = None,
    plots_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    """Forecast VaR over the test period, backtest it, and optionally save outputs.

    ``asset_returns`` should cover both the history before ``test_start`` (at
    least ``window`` days) and the test period itself.
    """
    asset_returns = asset_returns.dropna()
    test_index = asset_returns.loc[test_start:].index
    if test_index.empty:
        raise ValueError("Test period is empty.")

    forecasts = rolling_var_forecasts(
        asset_returns,
        weights,
        test_index,
        window=window,
        confidence_levels=confidence_levels,
        n_sims=n_sims,
        seed=seed,
    )
    report = backtest_var(forecasts, confidence_levels, significance)

    if verbose:
        print_backtest_report(report)

    plot_paths = []
    if results_dir is not None:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        forecasts.to_csv(results_dir / "var_forecasts.csv")
        report.to_csv(results_dir / "var_backtest.csv", index=False)
    if plots_dir is not None:
        for confidence in confidence_levels:
            plot_paths.append(plot_exceedances(
                forecasts, confidence,
                Path(plots_dir) / f"var_exceedances_{_level_tag(confidence)}.png",
            ))

    return {"forecasts": forecasts, "report": report, "plots": plot_paths}


def main(argv=None):
    from .backtest import run_backtest
    from .main import END_DATE, SENSEX_30, SENSEX_BENCHMARK, START_DATE, TRAIN_END, VAL_END

    parser = argparse.ArgumentParser(description="VaR engine and backtest for the Markowitz portfolio.")
    parser.add_argument("--window", type=int, default=250, help="rolling estimation window in trading days")
    parser.add_argument("--sims", type=int, default=10_000, help="Monte Carlo simulations per day")
    parser.add_argument("--seed", type=int, default=42, help="Monte Carlo random seed")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    result = run_backtest(
        SENSEX_30, START_DATE, END_DATE, TRAIN_END, VAL_END, SENSEX_BENCHMARK,
    )
    return run_var_analysis(
        result["asset_returns"],
        result["weights"],
        result["test_start"],
        window=args.window,
        n_sims=args.sims,
        seed=args.seed,
        results_dir=root / "results",
        plots_dir=root / "plots",
    )


if __name__ == "__main__":
    main()
