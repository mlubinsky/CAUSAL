"""From-scratch estimator/analysis helpers that consume SimResult.data only
(never .truth or .hidden) -- as opposed to tests/test_simulators.py, which
checks that each simulator's truth is recoverable by an oracle."""

import numpy as np
import pandas as pd
import pytest

import causalsim as cs


def test_its_wrong_intervention_date_biases_level_change():
    sim = cs.its_series(n=400, t0=250, level_change=-8.0, slope_change=0.0, ar1=0.0, noise=0.5, seed=20)
    correct = cs.estimate_its(sim.data, t0=250)
    wrong = cs.estimate_its(sim.data, t0=200)
    assert correct["level_change"] == pytest.approx(-8.0, abs=0.5)
    assert abs(wrong["level_change"] - (-8.0)) > 1.5


def test_its_autocorrelated_noise_understates_naive_se():
    sim = cs.its_series(n=2_000, t0=1_000, ar1=0.85, noise=1.0, seed=21)
    fit = cs.estimate_its(sim.data, t0=1_000)
    assert fit["se_hac"] > fit["se_ols"] * 1.3


def _diff_in_means(data: pd.DataFrame) -> tuple[float, float]:
    d1 = data.loc[data.D == 1, "Y"]
    d0 = data.loc[data.D == 0, "Y"]
    est = d1.mean() - d0.mean()
    se = np.sqrt(d1.var(ddof=1) / len(d1) + d0.var(ddof=1) / len(d0))
    return est, se


def test_monte_carlo_recovers_bias_and_coverage():
    result = cs.monte_carlo(
        generator=lambda seed: cs.confounded_binary(n=2_000, ate=2.0, confounding=0.0, seed=seed),
        estimator=_diff_in_means,
        n_runs=200,
    )
    assert abs(result["bias"]) < 0.1
    assert 0.85 <= result["coverage_95"] <= 1.0
    assert result["rmse"] > 0
    assert result["mean_se"] > 0
