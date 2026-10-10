# TODO

Items surviving an external review of this project (see discussion), rescoped as
concrete `causalsim` library features with a unit test apiece. Each item is
implemented test-first: a failing test is written against the not-yet-existing
behavior, then the minimal implementation is added to make it pass.

- [ ] **Latent confounder for causal discovery** — add a `latent` option to
  `linear_gaussian_sem()` that marginalizes out one node, inducing a spurious
  dependence between its children that is invisible to a graph built only from
  the DAG's explicit edges (breaks PC's causal-sufficiency assumption).

- [ ] **ITS intervention-timing / autocorrelation sensitivity** — add an
  `estimate_its()` helper to `causalsim` that fits the interrupted-time-series
  model from `its_series()` and demonstrate (via test) that (a) assuming the
  wrong intervention date biases the level-change estimate, and (b) ignoring
  AR(1) noise when computing standard errors understates them.

- [ ] **Reactive retail pricing** — add a `reactive` parameter to
  `retail_pricing()` so price responds to *lagged* purchases (a feedback/
  endogeneity channel distinct from the existing contemporaneous
  demand-shock confounding).

- [ ] **Instrument-validity violation in retail pricing** — add an
  `instrument_violation` parameter to `retail_pricing()` that correlates
  `cost` with the hidden demand shock, breaking the exclusion restriction
  that makes `cost` a valid instrument for `price`.

- [ ] **Reusable Monte Carlo benchmark utility** — add
  `causalsim.benchmark.monte_carlo(generator, estimator, n_runs, ...)`
  returning bias/RMSE/95%-CI coverage, generalizing the ad hoc Monte Carlo
  loops already duplicated across notebooks 01/03/05.

- [ ] **Backdoor adjustment-set identification** — add
  `causalsim.dag.backdoor_adjustment_sets(edges, treatment, outcome)`
  implementing the backdoor criterion over a DAG, tested against the
  collider/chain/fork structure in `EXAMPLE_EDGES`.

- [ ] **Full cross-elasticity matrix recovery** — add
  `causalsim.timeseries.estimate_elasticity_matrix(data)` that fits a
  log-log regression of each product's purchases on all three products'
  log-prices from `retail_pricing()` output, recovering the full 3×3 matrix
  (own- and cross-price terms), not just the own-price term each notebook
  currently uses.
