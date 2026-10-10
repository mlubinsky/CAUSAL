# Changelog

All notable changes to this project are recorded here, most recent first.

## Unreleased

- Added `anticipation_periods`/`anticipation_effect` to `did_panel()`: treated
  units can now start responding before the official `treat_period`, inflating
  the pre-period a naive DiD relies on. Exposed via `truth["anticipation_effect"]`
  and `meta["anticipation_start"]`. Covered by
  `test_did_anticipation_inflates_pre_period_gap`.
- Added a `latent` option to `linear_gaussian_sem()`: the named node is
  simulated but marginalized out of `data` into `hidden`, so its children stay
  spuriously correlated with no edge between them, violating the
  causal-sufficiency assumption PC relies on. Covered by
  `test_latent_confounder_induces_spurious_association`.
- Added `estimate_its()`: an OLS fit of the interrupted-time-series model with
  both an iid and a Newey-West (HAC) standard error for the level-change
  coefficient. Demonstrates that a mistimed intervention date biases the
  estimate, and that ignoring AR(1) noise understates its uncertainty.
  Covered by `tests/test_estimators.py`.
- Added a `reactive` parameter to `retail_pricing()`: price now responds to
  yesterday's purchases relative to their nominal rate ("mark down after weak
  sales"), a feedback/endogeneity channel through the lagged outcome, distinct
  from `price_endogeneity`'s contemporaneous demand-shock channel. The
  lagged-deviation signal is clipped to keep the feedback loop stable at
  typical elasticity magnitudes. Covered by
  `test_retail_pricing_reactive_price_follows_lagged_sales`.
- Added an `instrument_violation` parameter to `retail_pricing()`: correlates
  `cost` with the hidden demand shock, breaking the exclusion restriction that
  otherwise makes `cost` a valid instrument for `price`. Previously `cost` was
  exogenous by construction with no way to break it. Covered by
  `test_retail_pricing_instrument_violation_breaks_cost_exogeneity`.
