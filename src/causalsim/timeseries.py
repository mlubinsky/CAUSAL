"""Multivariate time-series simulators with known causal structure."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import SimResult, rng_from


def var_system(
    n: int = 1_000,
    coefs: dict[tuple[str, str, int], float] | None = None,
    names: tuple[str, ...] = ("visitors", "price", "purchases", "ads"),
    noise: float = 1.0,
    burn_in: int = 200,
    seed: int | None = 0,
) -> SimResult:
    """Stationary VAR with a user-specified lagged causal graph.

    coefs maps (cause, effect, lag) -> coefficient. Self-lags are included
    explicitly. The default graph:

        ads(t-1)       -> visitors(t)
        visitors(t-1)  -> purchases(t)
        price(t-1)     -> purchases(t)      (negative)
        purchases(t-2) -> price(t)          (feedback: pricing reacts to sales)

    truth["links"] is the set of (cause, effect, lag) with non-zero coefficient
    between DIFFERENT variables, i.e. what Granger tests should find.
    """
    rng = rng_from(seed)
    if coefs is None:
        coefs = {
            ("visitors", "visitors", 1): 0.5,
            ("price", "price", 1): 0.7,
            ("purchases", "purchases", 1): 0.3,
            ("ads", "ads", 1): 0.6,
            ("ads", "visitors", 1): 0.8,
            ("visitors", "purchases", 1): 0.6,
            ("price", "purchases", 1): -0.7,
            ("purchases", "price", 2): 0.25,
        }
    k = len(names)
    idx = {nm: i for i, nm in enumerate(names)}
    p = max(lag for (_, _, lag) in coefs)
    A = np.zeros((p, k, k))  # A[lag-1][effect, cause]
    for (c, e, lag), v in coefs.items():
        A[lag - 1, idx[e], idx[c]] = v

    _check_stationary(A)

    total = n + burn_in
    Y = np.zeros((total, k))
    E = rng.normal(scale=noise, size=(total, k))
    for t in range(p, total):
        Y[t] = sum(A[l] @ Y[t - l - 1] for l in range(p)) + E[t]
    Y = Y[burn_in:]

    data = pd.DataFrame(Y, columns=list(names))
    links = {(c, e, lag) for (c, e, lag), v in coefs.items() if c != e and v != 0}
    return SimResult(data=data, truth={"links": links, "coef_matrices": A, "max_lag": p})


def _check_stationary(A: np.ndarray) -> None:
    p, k, _ = A.shape
    comp = np.zeros((k * p, k * p))
    comp[:k, :] = np.hstack(list(A))
    if p > 1:
        comp[k:, :-k] = np.eye(k * (p - 1))
    rho = np.max(np.abs(np.linalg.eigvals(comp)))
    if rho >= 1:
        raise ValueError(f"VAR is not stationary (spectral radius {rho:.3f} >= 1)")


def retail_pricing(
    n_days: int = 730,
    elasticities: np.ndarray | None = None,
    price_endogeneity: float = 0.4,
    start: str = "2024-01-01",
    seed: int | None = 42,
) -> SimResult:
    """Daily store data shaped like a real retail CSV, with a known elasticity matrix.

    Columns: date, visitors, purch_1..3, price_1..3, cost_1..3

    - Weekly + yearly seasonality on traffic.
    - An unobserved, autocorrelated demand shock drives traffic, conversion,
      AND prices (price_endogeneity) -> naive regressions are biased.
    - cost_k shifts price_k but not demand -> valid instruments.
    - truth["elasticities"][i, j] = d log E[purch_i] / d log price_j.
    """
    rng = rng_from(seed)
    T = n_days
    t = np.arange(T)
    dow = t % 7
    season = 0.15 * np.sin(2 * np.pi * t / 365) + 0.10 * (dow >= 5)

    demand = np.zeros(T)
    for i in range(1, T):
        demand[i] = 0.7 * demand[i - 1] + rng.normal(0, 0.08)

    cost = rng.normal(0, 0.05, (T, 3))
    log_p = (
        np.log([4.0, 6.0, 9.0])
        + cost
        + price_endogeneity * demand[:, None]
        + rng.normal(0, 0.03, (T, 3))
    )
    log_visitors = 6.0 + season + 0.8 * demand + rng.normal(0, 0.05, T)

    E = (
        np.array([[-1.8, 0.5, 0.1], [0.4, -1.2, 0.0], [0.1, 0.0, -0.9]])
        if elasticities is None
        else np.asarray(elasticities)
    )
    base = np.log([0.05, 0.03, 0.02])
    rate = np.exp(base + (log_p - np.log([4.0, 6.0, 9.0])) @ E.T + 0.5 * demand[:, None])
    purch = rng.poisson(np.exp(log_visitors)[:, None] * rate)

    data = pd.DataFrame(
        {
            "date": pd.date_range(start, periods=T, freq="D"),
            "visitors": np.round(np.exp(log_visitors)).astype(int),
            **{f"purch_{k+1}": purch[:, k] for k in range(3)},
            **{f"price_{k+1}": np.exp(log_p[:, k]).round(2) for k in range(3)},
            **{f"cost_{k+1}": cost[:, k] for k in range(3)},
        }
    )
    return SimResult(
        data=data,
        truth={"elasticities": E, "traffic_elasticity": 1.0},
        hidden=pd.DataFrame({"demand_shock": demand}),
    )
