"""Panel and single-series intervention simulators: DiD, ITS, synthetic control."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import SimResult, rng_from


def did_panel(
    n_units: int = 200,
    n_periods: int = 20,
    treat_period: int = 10,
    effect: float = 3.0,
    treated_share: float = 0.5,
    selection_on_level: float = 2.0,
    trend_violation: float = 0.0,
    noise: float = 1.0,
    seed: int | None = 0,
) -> SimResult:
    """Long-format panel: y_it = a_i + l_t + effect*D_it + (trend_violation * t * treated_i) + e_it.

    selection_on_level : treated units have higher baseline levels. DiD handles this
                         (unit fixed effects absorb it); a naive post-period comparison does not.
    trend_violation    : treated units follow a different trend -> parallel trends broken.
    """
    rng = rng_from(seed)
    treated = (rng.random(n_units) < treated_share).astype(int)
    a = rng.normal(size=n_units) + selection_on_level * treated
    lam = np.cumsum(rng.normal(0.3, 0.5, size=n_periods))  # common time shocks

    unit = np.repeat(np.arange(n_units), n_periods)
    period = np.tile(np.arange(n_periods), n_units)
    tr = treated[unit]
    post = (period >= treat_period).astype(int)
    D = tr * post
    y = (
        a[unit]
        + lam[period]
        + effect * D
        + trend_violation * period * tr
        + rng.normal(scale=noise, size=unit.size)
    )
    data = pd.DataFrame({"unit": unit, "period": period, "treated": tr, "post": post, "D": D, "y": y})
    return SimResult(data=data, truth={"att": effect}, meta={"treat_period": treat_period})


def its_series(
    n: int = 400,
    t0: int = 250,
    level_change: float = -8.0,
    slope_change: float = 0.0,
    base_level: float = 100.0,
    base_slope: float = 0.05,
    weekly_amplitude: float = 5.0,
    ar1: float = 0.5,
    noise: float = 2.0,
    start: str = "2024-01-01",
    seed: int | None = 0,
) -> SimResult:
    """Daily series with an intervention at index t0 (interrupted time series).

        y_t = base + base_slope*t + level_change*post + slope_change*(t - t0)*post
              + weekly seasonality + AR(1) noise
    """
    rng = rng_from(seed)
    t = np.arange(n)
    post = (t >= t0).astype(int)
    eps = np.zeros(n)
    shocks = rng.normal(scale=noise, size=n)
    for i in range(1, n):
        eps[i] = ar1 * eps[i - 1] + shocks[i]
    dow = t % 7
    season = weekly_amplitude * (dow >= 5)  # weekend bump
    y = base_level + base_slope * t + level_change * post + slope_change * (t - t0) * post + season + eps
    dates = pd.date_range(start, periods=n, freq="D")
    data = pd.DataFrame({"date": dates, "y": y})
    return SimResult(
        data=data,
        truth={"level_change": level_change, "slope_change": slope_change},
        meta={"t0": t0, "intervention_date": dates[t0]},
    )


def synthetic_control_panel(
    n_donors: int = 20,
    n_periods: int = 60,
    t0: int = 40,
    effect: float = 5.0,
    n_factors: int = 2,
    noise: float = 0.5,
    seed: int | None = 0,
) -> SimResult:
    """Interactive fixed-effects (factor) model; unit 0 is treated from t0 on.

        y_it = mu_i + lambda_i . f_t + e_it

    The treated unit's (mu, lambda) is an exact convex combination of 3 donors'
    loadings, so a perfect synthetic control exists in population.
    """
    rng = rng_from(seed)
    f = np.cumsum(rng.normal(size=(n_periods, n_factors)), axis=0) * 0.7
    mu_d = rng.normal(10.0, 2.0, size=n_donors)
    lam_d = rng.normal(1.0, 0.7, size=(n_donors, n_factors))

    true_w = np.zeros(n_donors)
    idx = rng.choice(n_donors, size=3, replace=False)
    true_w[idx] = rng.dirichlet(np.ones(3))
    mu_t = true_w @ mu_d
    lam_t = true_w @ lam_d

    Yd = mu_d[None, :] + f @ lam_d.T + rng.normal(scale=noise, size=(n_periods, n_donors))
    y0_t = mu_t + f @ lam_t + rng.normal(scale=noise, size=n_periods)  # untreated potential outcome
    y_t = y0_t + effect * (np.arange(n_periods) >= t0)

    rows = [(0, t, y_t[t]) for t in range(n_periods)]
    rows += [(j + 1, t, Yd[t, j]) for j in range(n_donors) for t in range(n_periods)]
    data = pd.DataFrame(rows, columns=["unit", "period", "y"])
    data["treated"] = (data["unit"] == 0).astype(int)
    return SimResult(
        data=data,
        truth={"effect": effect, "weights": true_w, "y0_treated": y0_t},
        meta={"t0": t0, "treated_unit": 0},
    )
