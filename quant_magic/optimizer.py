from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def get_markowitz_weights(returns: pd.DataFrame) -> np.ndarray:
    """Optimize portfolio weights by maximizing Sharpe ratio under long-only constraints."""
    returns = returns.dropna()
    if returns.empty:
        raise ValueError("Returns DataFrame is empty.")

    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(returns.columns)

    def neg_sharpe(weights):
        p_ret = np.sum(mean_returns.to_numpy() * weights) * 252
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.to_numpy() * 252, weights)))
        return -p_ret / p_vol if p_vol != 0 else -1e9

    constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_guess = num_assets * [1.0 / num_assets]

    result = minimize(
        neg_sharpe,
        init_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000},
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    return result.x
