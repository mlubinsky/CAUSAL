"""Generic Monte Carlo evaluation of an estimator against a simulator's ground truth."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .core import SimResult


def monte_carlo(
    generator: Callable[[int], SimResult],
    estimator: Callable[[pd.DataFrame], tuple[float, float]],
    n_runs: int = 200,
    truth_key: str = "ate",
    seed0: int = 0,
) -> dict:
    """Run `estimator` on `n_runs` independent draws from `generator` and summarize.

    generator(seed) -> SimResult. estimator(data) -> (estimate, se); the estimator
    only ever sees `.data` -- `.truth` is read here, after fitting, for evaluation
    only. This generalizes the ad hoc Monte Carlo loops duplicated across several
    notebooks into one reusable utility.

    Returns a dict with bias, rmse, 95% CI coverage (estimate +/- 1.96*se), the
    mean reported se, and the raw per-run results.
    """
    rows = []
    for i in range(n_runs):
        sim = generator(seed0 + i)
        estimate, se = estimator(sim.data)
        truth = sim.truth[truth_key]
        rows.append(
            {
                "estimate": estimate,
                "se": se,
                "truth": truth,
                "error": estimate - truth,
                "covered": (estimate - 1.96 * se) <= truth <= (estimate + 1.96 * se),
            }
        )
    df = pd.DataFrame(rows)
    return {
        "bias": float(df["error"].mean()),
        "rmse": float(np.sqrt(np.mean(df["error"] ** 2))),
        "coverage_95": float(df["covered"].mean()),
        "mean_se": float(df["se"].mean()),
        "runs": df,
    }
