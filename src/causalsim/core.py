"""Shared container for simulated data plus its ground truth."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SimResult:
    """What every simulator returns.

    data   : the observed data an analyst would see (hidden variables excluded).
    truth  : the ground-truth quantities the estimators should recover.
    hidden : variables that exist in the data-generating process but are NOT
             observed (e.g. an unobserved confounder). Useful for debugging and
             for "oracle" comparisons; never pass these to an estimator.
    """

    data: pd.DataFrame
    truth: dict[str, Any]
    hidden: pd.DataFrame | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # compact, notebook-friendly
        t = ", ".join(f"{k}={_short(v)}" for k, v in self.truth.items())
        return f"SimResult(n={len(self.data)}, cols={list(self.data.columns)}, truth={{{t}}})"


def _short(v: Any) -> str:
    if isinstance(v, (float, np.floating)):
        return f"{v:.4g}"
    if isinstance(v, np.ndarray):
        return f"array{v.shape}"
    if isinstance(v, (list, tuple, set)) and len(v) > 4:
        return f"{type(v).__name__}[{len(v)}]"
    return repr(v)


def rng_from(seed: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
