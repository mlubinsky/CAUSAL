# Changelog

All notable changes to this project are recorded here, most recent first.

## Unreleased

- Added `anticipation_periods`/`anticipation_effect` to `did_panel()`: treated
  units can now start responding before the official `treat_period`, inflating
  the pre-period a naive DiD relies on. Exposed via `truth["anticipation_effect"]`
  and `meta["anticipation_start"]`. Covered by
  `test_did_anticipation_inflates_pre_period_gap`.
