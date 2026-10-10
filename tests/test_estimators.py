"""From-scratch estimator/analysis helpers that consume SimResult.data only
(never .truth or .hidden) -- as opposed to tests/test_simulators.py, which
checks that each simulator's truth is recoverable by an oracle."""

import numpy as np
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
