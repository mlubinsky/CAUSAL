"""Sanity checks: each simulator's truth is recoverable by an 'oracle' that is
allowed to see what a real analyst cannot (randomization, hidden variables)."""

import numpy as np
import pytest

import causalsim as cs


def ols(X, y):
    X = np.column_stack([np.ones(len(y)), X])
    return np.linalg.lstsq(X, y, rcond=None)[0]


def test_randomized_experiment_difference_in_means():
    sim = cs.confounded_binary(n=50_000, ate=2.0, confounding=0.0, seed=1)
    d = sim.data
    diff = d.loc[d.D == 1, "Y"].mean() - d.loc[d.D == 0, "Y"].mean()
    assert diff == pytest.approx(2.0, abs=0.05)


def test_confounding_biases_naive_estimate():
    sim = cs.confounded_binary(n=50_000, ate=2.0, confounding=1.0, seed=1)
    d = sim.data
    diff = d.loc[d.D == 1, "Y"].mean() - d.loc[d.D == 0, "Y"].mean()
    assert abs(diff - 2.0) > 1.0


def test_heterogeneous_effect_keeps_ate():
    sim = cs.confounded_binary(n=100_000, ate=2.0, effect_heterogeneity=1.5, seed=2)
    assert sim.truth["cate"].mean() == pytest.approx(2.0, abs=0.02)
    assert sim.truth["att"] > 2.2  # treated units have higher X1 -> larger effects


def test_iv_oracle_with_hidden_confounder():
    sim = cs.iv_data(n=50_000, effect=1.5, seed=3)
    d, U = sim.data, sim.hidden["U"]
    b = ols(np.column_stack([d.D, d.W, U]), d.Y)
    assert b[1] == pytest.approx(1.5, abs=0.02)


def test_plr_oracle():
    sim = cs.plr_nonlinear(n=20_000, theta=0.5, seed=4)
    assert sim.truth["theta"] == 0.5
    assert sim.data.shape == (20_000, 12)


def test_did_two_by_two():
    sim = cs.did_panel(n_units=2_000, effect=3.0, seed=5)
    d = sim.data
    m = d.groupby(["treated", "post"])["y"].mean()
    did = (m[1, 1] - m[1, 0]) - (m[0, 1] - m[0, 0])
    assert did == pytest.approx(3.0, abs=0.15)


def test_its_shapes_and_truth():
    sim = cs.its_series(seed=6)
    assert len(sim.data) == 400
    assert sim.meta["intervention_date"] == sim.data.date.iloc[sim.meta["t0"]]


def test_synthetic_control_true_weights_reproduce_counterfactual():
    sim = cs.synthetic_control_panel(noise=0.0, seed=7)
    wide = sim.data.pivot(index="period", columns="unit", values="y")
    donors = wide.drop(columns=0).to_numpy()
    synth = donors @ sim.truth["weights"]
    np.testing.assert_allclose(synth, sim.truth["y0_treated"], atol=1e-8)
    assert sim.truth["weights"].sum() == pytest.approx(1.0)


def test_var_ols_recovers_coefficients():
    sim = cs.var_system(n=20_000, seed=8)
    d = sim.data.to_numpy()
    p, A = sim.truth["max_lag"], sim.truth["coef_matrices"]
    Y = d[p:]
    X = np.hstack([d[p - l - 1 : len(d) - l - 1] for l in range(p)])
    B = np.linalg.lstsq(np.column_stack([np.ones(len(Y)), X]), Y, rcond=None)[0][1:]
    est = B.T.reshape(d.shape[1], p, d.shape[1]).transpose(1, 0, 2)
    np.testing.assert_allclose(est, A, atol=0.03)


def test_var_rejects_explosive_system():
    with pytest.raises(ValueError):
        cs.var_system(coefs={("a", "a", 1): 1.1}, names=("a",))


def test_sem_regression_recovers_weights():
    sim = cs.linear_gaussian_sem(n=50_000, seed=9)
    d, W, nodes = sim.data, sim.truth["weights"], sim.truth["nodes"]
    c = nodes.index("C")
    parents = np.flatnonzero(W[:, c])
    b = ols(d.iloc[:, parents].to_numpy(), d.iloc[:, c].to_numpy())
    np.testing.assert_allclose(b[1:], W[parents, c], atol=0.03)


def test_random_dag_is_acyclic():
    edges = cs.random_dag(n_nodes=12, edge_prob=0.4, seed=10)
    sim = cs.linear_gaussian_sem(n=100, edges=edges, seed=10)  # raises on a cycle
    assert sim.truth["adjacency"].sum() == len(edges)


def test_retail_pricing_columns():
    sim = cs.retail_pricing(n_days=100)
    assert list(sim.data.columns) == [
        "date", "visitors", "purch_1", "purch_2", "purch_3",
        "price_1", "price_2", "price_3", "cost_1", "cost_2", "cost_3",
    ]
    assert sim.truth["elasticities"].shape == (3, 3)
