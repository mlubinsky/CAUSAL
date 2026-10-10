"""Cross-sectional simulators: binary treatment, instruments, partially linear models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import SimResult, rng_from, sigmoid


def confounded_binary(
    n: int = 5_000,
    ate: float = 2.0,
    n_covariates: int = 3,
    confounding: float = 1.0,
    effect_heterogeneity: float = 0.0,
    nonlinear_outcome: bool = False,
    nonlinear_propensity: bool = False,
    hidden_confounding: float = 0.0,
    seed: int | None = 0,
) -> SimResult:
    """Binary treatment D, outcome Y, observed confounders X1..Xp.

    Knobs (each one breaks a specific estimator assumption):
      confounding          strength of X -> D. 0 gives a randomized experiment.
                           Large values push propensities toward 0/1 (poor overlap).
      effect_heterogeneity CATE(x) = ate + h * X1. X1 has mean 0, so ATE stays `ate`.
      nonlinear_outcome    adds non-linear terms to E[Y|X] (breaks linear outcome models).
      nonlinear_propensity adds non-linear terms to the propensity (breaks logistic models).
      hidden_confounding   strength of an UNOBSERVED U affecting both D and Y.
                           No estimator using only (X, D, Y) can fix this.
    """
    rng = rng_from(seed)
    X = rng.normal(size=(n, n_covariates))
    U = rng.normal(size=n)

    g = np.linspace(1.0, 0.3, n_covariates)  # propensity coefficients
    b = np.linspace(1.5, 0.5, n_covariates)  # outcome coefficients

    logit = confounding * (X @ g) + hidden_confounding * U
    if nonlinear_propensity:
        logit = logit + confounding * (X[:, 0] ** 2 - 1.0) - confounding * np.abs(X[:, 1])
    e = sigmoid(logit)
    T = rng.binomial(1, e)

    mu0 = X @ b + hidden_confounding * 2.0 * U
    if nonlinear_outcome:
        mu0 = mu0 + 2.0 * np.sin(2.0 * X[:, 0]) + (X[:, 1] ** 2 - 1.0) + X[:, 0] * X[:, 2 % n_covariates]
    cate = ate + effect_heterogeneity * X[:, 0]
    Y = mu0 + cate * T + rng.normal(scale=1.0, size=n)

    cols = {f"X{j+1}": X[:, j] for j in range(n_covariates)}
    data = pd.DataFrame({**cols, "D": T, "Y": Y})
    truth = {
        "ate": ate,
        # ATT differs from ATE when effects are heterogeneous AND treatment depends on X1
        "att": float(np.mean(cate[T == 1])),
        "cate": cate,
        "propensity": e,
    }
    return SimResult(data=data, truth=truth, hidden=pd.DataFrame({"U": U}))


def iv_data(
    n: int = 5_000,
    effect: float = 1.5,
    instrument_strength: float = 1.0,
    endogeneity: float = 1.0,
    exclusion_violation: float = 0.0,
    seed: int | None = 0,
) -> SimResult:
    """Continuous endogenous treatment D with instrument Z and observed control W.

        U  ~ N(0,1)                         (unobserved confounder)
        D  = pi*Z + 0.5*W + endogeneity*U + noise
        Y  = effect*D + 1.0*W + 2*U + exclusion_violation*Z + noise

    instrument_strength (pi): small values -> weak instrument.
    exclusion_violation      : Z affects Y directly -> IV assumption broken.
    """
    rng = rng_from(seed)
    Z = rng.normal(size=n)
    W = rng.normal(size=n)
    U = rng.normal(size=n)
    D = instrument_strength * Z + 0.5 * W + endogeneity * U + rng.normal(size=n)
    Y = effect * D + 1.0 * W + 2.0 * U + exclusion_violation * Z + rng.normal(size=n)
    data = pd.DataFrame({"Z": Z, "W": W, "D": D, "Y": Y})
    truth = {"effect": effect, "first_stage": instrument_strength}
    return SimResult(data=data, truth=truth, hidden=pd.DataFrame({"U": U}))


def plr_nonlinear(
    n: int = 2_000,
    theta: float = 0.5,
    n_covariates: int = 10,
    confounding: float = 1.0,
    seed: int | None = 0,
) -> SimResult:
    """Partially linear model with non-linear confounding (the DML setting).

        D = m(X) + v,          m non-linear in X
        Y = theta*D + g(X) + e g non-linear in X

    Linear regression of Y on (D, X) is biased because g and m are non-linear;
    double ML with flexible learners recovers theta.
    """
    rng = rng_from(seed)
    X = rng.normal(size=(n, n_covariates))
    m = confounding * (np.sin(X[:, 0]) * 2.0 + 0.5 * X[:, 1] ** 2 + np.tanh(X[:, 2]))
    g = confounding * (np.cos(X[:, 0]) * 2.0 + 0.8 * X[:, 1] ** 2 + np.exp(-np.abs(X[:, 3])) * 2.0)
    D = m + rng.normal(size=n)
    Y = theta * D + g + rng.normal(size=n)
    cols = {f"X{j+1}": X[:, j] for j in range(n_covariates)}
    data = pd.DataFrame({**cols, "D": D, "Y": Y})
    return SimResult(data=data, truth={"theta": theta})
