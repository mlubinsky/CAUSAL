"""causalsim: data generators with known ground truth for learning causal inference.

Every simulator returns a SimResult with:
    .data    observed DataFrame
    .truth   dict of true causal quantities
    .hidden  unobserved variables (never give these to an estimator)
"""

from .benchmark import monte_carlo
from .core import SimResult
from .cross_section import confounded_binary, iv_data, plr_nonlinear
from .dag import EXAMPLE_EDGES, backdoor_adjustment_sets, linear_gaussian_sem, random_dag
from .panel import did_panel, estimate_its, its_series, synthetic_control_panel
from .timeseries import retail_pricing, var_system

__all__ = [
    "SimResult",
    "confounded_binary",
    "iv_data",
    "plr_nonlinear",
    "did_panel",
    "its_series",
    "estimate_its",
    "synthetic_control_panel",
    "var_system",
    "retail_pricing",
    "linear_gaussian_sem",
    "random_dag",
    "EXAMPLE_EDGES",
    "backdoor_adjustment_sets",
    "monte_carlo",
]
