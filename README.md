# causal-from-scratch

Learn causal inference by implementing each core estimator yourself, on simulated data where
the true answer is known, and then checking your implementation against an established library.

Every notebook follows the same loop:

1. **Simulate** data with a known causal effect (`causalsim`).
2. **Implement** the estimator from scratch in NumPy (10–60 lines).
3. **Check** that it recovers the truth.
4. **Compare** with the reference library. The notebook `assert`s an exact (or solver-tolerance) match.
5. **Break it**: violate one assumption at a time and watch the estimate fail.

## Notebooks

| # | Technique | You implement | Checked against |
|---|---|---|---|
| 01 | Confounding & regression adjustment | OLS + classical/HC1 SEs, g-computation | `statsmodels` |
| 02 | Propensity scores | Newton–Raphson logit, Horvitz–Thompson & Hájek IPW, NN matching, ESS | `statsmodels`, `dowhy` |
| 03 | Doubly robust (AIPW) | cross-fitted AIPW, influence-function SE, coverage check | `doubleml.DoubleMLIRM` |
| 04 | Instrumental variables | Wald, 2SLS + classical/robust SEs, first-stage F, weak-IV Monte Carlo | `linearmodels.IV2SLS` |
| 05 | Double ML (partially linear) | cross-fitted partialling-out, orthogonal-score SE | `doubleml.DoubleMLPLR` |
| 06 | Difference-in-differences | 2×2 DiD, TWFE by double demeaning, cluster SEs, event study | `linearmodels.PanelOLS` |
| 07 | ITS & synthetic control | segmented regression + Newey–West, constrained weights, placebo inference | `statsmodels` HAC, `pysyncon` |
| 08 | VAR & Granger causality | VAR by OLS, AIC lag selection, Granger F-test, impulse responses | `statsmodels` VAR |
| 09 | PC algorithm | Fisher-z CI test, PC-stable skeleton, v-structures, Meek rules | `causal-learn` |

01–03 build on each other (outcome model → treatment model → both). 04, 06, 07 and 08 map most directly onto
price/sales time series. Each notebook ends with exercises.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .               # makes `import causalsim` work from the notebooks
pytest                         # simulator sanity checks (~1 s)
jupyter lab notebooks/
```

Run every notebook headlessly. A notebook fails if any from-scratch result disagrees with its library:

```bash
python scripts/run_notebooks.py            # all (~3 min; notebook 05 is the slow one)
python scripts/run_notebooks.py 04 06      # a subset
```

Developed with Python 3.13, NumPy 2.5, pandas 3.0, statsmodels 0.15, linearmodels 7.0, DoubleML 0.11,
DoWhy 0.14, causal-learn 0.1.4, pysyncon 1.7.

## The simulator package: `causalsim`

Every generator returns a `SimResult`:

```python
import causalsim as cs

sim = cs.confounded_binary(n=5_000, ate=2.0, confounding=1.0, hidden_confounding=0.0, seed=0)
sim.data     # what an analyst observes (DataFrame)
sim.truth    # {'ate': 2.0, 'att': ..., 'cate': array, 'propensity': array}
sim.hidden   # unobserved variables. Never pass these to an estimator.
```

| Generator | Setting | Knobs that break assumptions |
|---|---|---|
| `confounded_binary` | binary treatment, observed confounders | `confounding` (overlap), `nonlinear_outcome`, `nonlinear_propensity`, `effect_heterogeneity`, `hidden_confounding` |
| `iv_data` | endogenous treatment + instrument | `instrument_strength`, `endogeneity`, `exclusion_violation` |
| `plr_nonlinear` | partially linear model, non-linear confounding | `confounding` |
| `did_panel` | panel with a treatment date | `selection_on_level` (harmless), `trend_violation` |
| `its_series` | daily series with an intervention | `level_change`, `slope_change`, `ar1`, `weekly_amplitude` |
| `synthetic_control_panel` | factor model, one treated unit | `n_donors`, `n_factors`, `noise` |
| `var_system` | VAR with a user-specified lagged graph | `coefs={(cause, effect, lag): value}` |
| `linear_gaussian_sem`, `random_dag` | static DAGs for discovery | `edges`, `weight_range` |
| `retail_pricing` | daily store data: visitors, purchases, prices, costs | `elasticities` (3×3 truth), `price_endogeneity` |

`retail_pricing` produces a CSV-shaped frame like a real store export, with a hidden demand shock that
drives prices (so naive regressions are biased) and cost shifters that work as instruments. Several
exercises use it as a capstone.

Naming note: the treatment column is `D`, never `T`, because `df.T` is pandas' transpose.

## Library quirks the checks surfaced

Matching a library *exactly* forces you to learn what it actually does:

- **DoWhy** clips propensity scores to [0.05, 0.95] by default, and its `ips_weight` scheme is the normalized
  (Hájek) estimator for the ATE.
- **DoubleML** truncates propensities at 0.01 by default; pass identical folds with `set_sample_splitting`
  to reproduce it to machine precision.
- **linearmodels** `PanelOLS` clustered SEs need `group_debias=True` to match the Stata-style small-sample factor.
- **statsmodels** `VAR.test_causality` reports the same F statistic as a single-equation test but with
  system-wide denominator degrees of freedom, so the p-values differ.
- **pysyncon** rescales every predictor by its cross-unit standard deviation before optimizing the weights.
- **pandas 3**: averaging an object column that mixes `NaN` and `bool` gives wrong numbers. Store indicators as floats.

## Layout

```
src/causalsim/        simulators (cross_section, panel, timeseries, dag) + plotting style
notebooks/            01–09, saved with outputs
tests/                simulator sanity checks (oracle estimators that may see hidden variables)
scripts/              run_notebooks.py
```

## Suggested reading alongside

- *Causal Inference for the Brave and True*, Matheus Facure (free, Python)
- *Causal Inference: The Mixtape*, Scott Cunningham (free; IV, DiD, synthetic control)
- *Causal Inference: What If*, Hernán & Robins (free; identification, g-methods)
- Chernozhukov et al. (2018), "Double/debiased machine learning for treatment and structural parameters"
