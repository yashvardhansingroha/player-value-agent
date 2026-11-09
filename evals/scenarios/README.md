# Scenario files (expectations written BEFORE any data generation or agent run)

Each `*.json` file is one planted ground-truth scenario. Everything under `expected` is the answer a correct
review should give, written from the design intent, not from looking at generated data. These files are read only
by evaluation and test code. The decision-facing data layer, tools and agent never read this folder.

Action vocabulary: `increase_test`, `hold`, `decrease`, `insufficient_evidence`.
- `ideal_actions`: the best answers. Abstention is correct only when `insufficient_evidence` is ideal.
- `acceptable_actions`: safe but not ideal.
- `forbidden_actions`: counted as wrong decisions.

`calibration_intents` are generator checks: after the generator is built, the dev-seed data must satisfy them. If it does
not, the GENERATOR is fixed, not the intent. A test reads the thresholds from these files.

Metric definitions used by `calibration_intents` (all on the dev seed, using the full stored data, which is allowed for
generator validation because it is not a decision):
- `avg_d30_roas_over_target`: for the channel, over cohorts installed on days 0..184, `sum(D30 net revenue) / sum(spend)`
  divided by `target`, where `target = sum(D30 net) / sum(D180 net)` over the same cohorts (payback multiple 1.0).
- `true_step20_marginal_roas_over_target`: from hidden truth, the extra D30 net revenue of raising daily spend from its
  base to 1.2 x base, divided by the extra spend, divided by the same `target`. This is an APPROXIMATE CALIBRATION PROXY
  (it assumes value per install is proportional to player quality), not exact simulation truth, and it must never be used as
  the oracle for scoring future experiments.
- `d30_to_d3_multiplier_over_peer_median`: channel's `sum(D30 net) / sum(D3 net)` divided by the median of the other paid channels'.
- `d3_net_revenue_per_install_rank`: rank of the channel's D3 net revenue per install among the 5 paid channels (1 = highest).
- `outage_event_drop_fraction`: 1 - (active player-days per install-day-weighted player in the window) / (same over the 14 days before it).
- `outage_install_change_abs`: absolute relative change of mean daily installs, window vs the 14 days before.
- `cpi_spike_ratio_to_trailing_median`: mean CPI in the window / median daily CPI of the 28 days before it.
- `min_report_lag_days`, `other_channels_max_report_lag_days`: smallest / largest `reported_at - event_date` in days.
- `naive_visible_d3_revenue_fraction`: for the channel at as_of day 301, D0-D3 net revenue visible for cohorts installed
  days 293..297 (amended from 285..291, see delayed_reporting.json) divided by the same quantity at as_of day 336.
- `outage_event_drop_fraction` (implementation detail): per event day, active player-days divided by players installed in
  the previous 30 days (inclusive), averaged over the window and over the 14 days before it.
- `window_installs_per_day_max`: maximum daily installs inside the window.
- `whale_cohort_d3_net_over_normal_median`: D0-D3 net revenue of the whale cohort / median of the channel's other cohorts.
- `iap_per_active_day_post_over_pre`, `ad_ecpm_post_over_pre`: ratio of the mean over days 280..336 to the mean over days 224..279.
