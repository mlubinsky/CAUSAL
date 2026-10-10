"""Static DAG simulators for causal discovery (linear Gaussian SEMs)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import SimResult, rng_from

# A small DAG with a collider, a chain and a fork -- enough to see every PC step:
#   A -> C <- B      (v-structure: PC can orient it)
#   C -> D -> E      (chain)
#   C -> F           (fork from C)
EXAMPLE_EDGES = [("A", "C"), ("B", "C"), ("C", "D"), ("D", "E"), ("C", "F")]


def linear_gaussian_sem(
    n: int = 5_000,
    edges: list[tuple[str, str]] | None = None,
    weight_range: tuple[float, float] = (0.5, 1.5),
    noise: float = 1.0,
    seed: int | None = 0,
    latent: str | None = None,
) -> SimResult:
    """Sample from X_j = sum_{i in pa(j)} w_ij X_i + N(0, noise^2).

    truth["adjacency"][i, j] = 1 iff i -> j. truth["edges"] lists (parent, child).

    latent : if given, this node is simulated as usual but then marginalized out
             of `data` into `hidden`. Its children stay correlated through it even
             though no edge connects them to each other -- the causal-sufficiency
             assumption PC relies on (no unmeasured common causes) is violated.
    """
    rng = rng_from(seed)
    edges = EXAMPLE_EDGES if edges is None else edges
    nodes = sorted({v for e in edges for v in e})
    idx = {v: i for i, v in enumerate(nodes)}
    k = len(nodes)
    W = np.zeros((k, k))
    for a, b in edges:
        W[idx[a], idx[b]] = rng.uniform(*weight_range) * rng.choice([-1, 1])

    order = _topological_order(W)
    X = np.zeros((n, k))
    for j in order:
        X[:, j] = X @ W[:, j] + rng.normal(scale=noise, size=n)

    adj = (W != 0).astype(int)
    full = pd.DataFrame(X, columns=nodes)
    data, hidden = full, None
    if latent is not None:
        hidden = full[[latent]].copy()
        data = full.drop(columns=[latent])
    return SimResult(
        data=data,
        truth={"adjacency": adj, "edges": list(edges), "weights": W, "nodes": nodes, "latent": latent},
        hidden=hidden,
    )


def random_dag(
    n_nodes: int = 8,
    edge_prob: float = 0.25,
    seed: int | None = 0,
) -> list[tuple[str, str]]:
    """Random DAG edges (Erdos-Renyi on a random ordering)."""
    rng = rng_from(seed)
    names = [f"V{i}" for i in range(n_nodes)]
    perm = rng.permutation(n_nodes)
    return [
        (names[perm[i]], names[perm[j]])
        for i in range(n_nodes)
        for j in range(i + 1, n_nodes)
        if rng.random() < edge_prob
    ]


def _topological_order(W: np.ndarray) -> list[int]:
    k = W.shape[0]
    indeg = (W != 0).sum(axis=0).astype(int)
    ready = [j for j in range(k) if indeg[j] == 0]
    order = []
    while ready:
        j = ready.pop()
        order.append(j)
        for c in np.flatnonzero(W[j]):
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(c)
    if len(order) != k:
        raise ValueError("edges contain a cycle")
    return order
