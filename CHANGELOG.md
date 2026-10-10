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
