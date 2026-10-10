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


def estimate_elasticity_matrix(data: pd.DataFrame, n_products: int = 3) -> np.ndarray:
    """Log-log OLS: log(purch_k + 1) ~ const + log(visitors) + log(price_1..n).

    Recovers the full elasticity matrix E[k, j] = d log E[purch_k] / d log price_j
    -- own- AND cross-price terms -- rather than the single own-price term each
    notebook currently pulls out in isolation. Consistent when prices are
    exogenous (price_endogeneity = reactive = instrument_violation = 0); biased
    otherwise, like any naive regression on confounded data.
    """
    log_price = np.column_stack([np.log(data[f"price_{j+1}"]) for j in range(n_products)])
    log_visitors = np.log(data["visitors"]).to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(data)), log_visitors, log_price])
    E = np.zeros((n_products, n_products))
    for k in range(n_products):
        y = np.log1p(data[f"purch_{k+1}"].to_numpy(dtype=float))
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        E[k, :] = beta[2:]
    return E


def retail_pricing(
    n_days: int = 730,
    elasticities: np.ndarray | None = None,
    price_endogeneity: float = 0.4,
    reactive: float = 0.0,
    instrument_violation: float = 0.0,
    start: str = "2024-01-01",
    seed: int | None = 42,
) -> SimResult:
    """Daily store data shaped like a real retail CSV, with a known elasticity matrix.

    Columns: date, visitors, purch_1..3, price_1..3, cost_1..3

    - Weekly + yearly seasonality on traffic.
    - An unobserved, autocorrelated demand shock drives traffic, conversion,
      AND prices (price_endogeneity) -> naive regressions are biased.
    - cost_k shifts price_k but not demand -> valid instruments, AS LONG AS
      instrument_violation is 0.
    - truth["elasticities"][i, j] = d log E[purch_i] / d log price_j.

    reactive : feedback strength from YESTERDAY's purchases (relative to their
               nominal, no-shock rate) to TODAY's log price -- "mark down after
               weak sales". This is endogeneity through the lagged outcome,
               distinct from price_endogeneity's contemporaneous demand-shock
               channel, and from cost, which stays a valid instrument either way.
    instrument_violation : correlates cost with the hidden demand shock, breaking
               the exclusion restriction that makes cost a valid instrument for
               price (cost would then be associated with an unobserved driver of
               purchases, not just with price).
    """
    rng = rng_from(seed)
    T = n_days
    t = np.arange(T)
    dow = t % 7
    season = 0.15 * np.sin(2 * np.pi * t / 365) + 0.10 * (dow >= 5)

    demand = np.zeros(T)
    for i in range(1, T):
        demand[i] = 0.7 * demand[i - 1] + rng.normal(0, 0.08)

    cost = rng.normal(0, 0.05, (T, 3)) + instrument_violation * demand[:, None]
    base_price = np.log([4.0, 6.0, 9.0])
    price_noise = rng.normal(0, 0.03, (T, 3))
    log_visitors = 6.0 + season + 0.8 * demand + rng.normal(0, 0.05, T)
    visitors = np.exp(log_visitors)

    E = (
        np.array([[-1.8, 0.5, 0.1], [0.4, -1.2, 0.0], [0.1, 0.0, -0.9]])
        if elasticities is None
        else np.asarray(elasticities)
    )
    base = np.log([0.05, 0.03, 0.02])

    if reactive == 0.0:
        log_p = base_price + cost + price_endogeneity * demand[:, None] + price_noise
        rate = np.exp(base + (log_p - base_price) @ E.T + 0.5 * demand[:, None])
        purch = rng.poisson(visitors[:, None] * rate)
    else:
        nominal = visitors[:, None] * np.exp(base)
        log_p = np.zeros((T, 3))
        purch = np.zeros((T, 3), dtype=int)
        dev_lag = np.zeros(3)
        for i in range(T):
            log_p[i] = base_price + cost[i] + price_endogeneity * demand[i] + reactive * dev_lag + price_noise[i]
            rate_i = np.exp(base + (log_p[i] - base_price) @ E.T + 0.5 * demand[i])
            purch[i] = rng.poisson(visitors[i] * rate_i)
            dev_lag = np.clip(np.log(purch[i] + 1) - np.log(nominal[i] + 1), -1.0, 1.0)

    data = pd.DataFrame(
        {
            "date": pd.date_range(start, periods=T, freq="D"),
            "visitors": np.round(visitors).astype(int),
            **{f"purch_{k+1}": purch[:, k] for k in range(3)},
            **{f"price_{k+1}": np.exp(log_p[:, k]).round(2) for k in range(3)},
            **{f"cost_{k+1}": cost[:, k] for k in range(3)},
        }
    )
    return SimResult(
        data=data,
        truth={
            "elasticities": E,
            "traffic_elasticity": 1.0,
            "reactive": reactive,
            "instrument_violation": instrument_violation,
        },
        hidden=pd.DataFrame({"demand_shock": demand}),
    )
