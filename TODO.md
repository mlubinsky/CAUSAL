# TODO

Items surviving an external review of this project (see discussion), rescoped as
concrete `causalsim` library features with a unit test apiece. Each item is
implemented test-first: a failing test is written against the not-yet-existing
behavior, then the minimal implementation is added to make it pass.

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
