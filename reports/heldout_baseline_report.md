# Held-out evaluation of the frozen deterministic baseline

**What this is.** An evaluation of the FROZEN DETERMINISTIC RULES-ONLY BASELINE on three additional synthetic seeds. It validates nothing about any live agent: no model was called, there is no agent-vs-baseline comparison, and live Gemini verification remains PENDING (blocked by a recorded billing state, HTTP 402). These seeds have now been used for this evaluation, so they can NOT later be described as untouched holdouts for an agent. All data is synthetic; the ledger is a sandbox; experiments are simulated.

## Run status

| seed | generate | backtest | tie_checks | evaluate | freeze | commit |
|---|---|---|---|---|---|---|
| 9101 | completed | completed | completed | completed | baseline-freeze-v1 | abd7625e71 |
| 9102 | completed | completed | completed | completed | baseline-freeze-v1 | abd7625e71 |
| 9103 | completed | completed | completed | completed | baseline-freeze-v1 | abd7625e71 |

## Per-seed results

### Seed 9101

Artifact hashes (sha256, from provenance.json): decision_db `bdc25a086193`, hidden_truth `984517758644`, decision_db_at_backtest_start `bdc25a086193`, forecast_store `00220884508b`

#### Forecasting (scored against D30 outcomes in storage; nominal interval coverage 0.80)

| level | group | n | wape | bias | n_with_interval | coverage | interval_width_over_forecast_mean | interval_width_over_forecast_median |
|---|---|---|---|---|---|---|---|---|
| cohort | overall | 1329 | 0.267 | -0.031 | 1004 | 0.783 | 0.796 | 0.824 |
| weekly_summary | overall | 215 | 0.147 | -0.031 | 140 | 0.671 | 0.402 | 0.449 |
| cohort | channel applovin | 278 | 0.270 | -0.044 | 213 | 0.742 | 0.752 | 0.780 |
| cohort | channel google_uac | 256 | 0.291 | 0.003 | 197 | 0.777 | 0.956 | 0.963 |
| cohort | channel meta | 282 | 0.295 | -0.042 | 215 | 0.763 | 0.957 | 0.937 |
| cohort | channel tiktok | 251 | 0.119 | 0.030 | 188 | 0.894 | 0.379 | 0.380 |
| cohort | channel unity | 262 | 0.315 | -0.069 | 191 | 0.749 | 0.912 | 0.895 |
| weekly_summary | channel applovin | 43 | 0.159 | -0.044 | 28 | 0.643 | 0.523 | 0.534 |
| weekly_summary | channel google_uac | 43 | 0.161 | 0.003 | 28 | 0.607 | 0.416 | 0.416 |
| weekly_summary | channel meta | 43 | 0.150 | -0.042 | 28 | 0.536 | 0.406 | 0.435 |
| weekly_summary | channel tiktok | 43 | 0.060 | 0.030 | 28 | 0.857 | 0.154 | 0.144 |
| weekly_summary | channel unity | 43 | 0.184 | -0.069 | 28 | 0.714 | 0.509 | 0.519 |

Accounting (cohorts): 1329 issued, 1329 scored, 0 pending, 324 withheld (below_min_installs: 35, insufficient_training_history: 145, low_activity: 7, revenue_spike: 137); share of matured D30 revenue withheld 0.353.

#### Policy and abstentions - PRIMARY (all scored rows, as registered)

48 scored channel-reviews: ideal 34, acceptable 11, forbidden 3, unlisted 0 (ideal 0.708, acceptable-or-better 0.938, forbidden 0.062).

| scenario | n_scored | ideal | acceptable | forbidden | unlisted |
|---|---|---|---|---|---|
| cpi_spike | 2 | 1 | 1 | 0 | 0 |
| delayed_reporting | 14 | 11 | 0 | 3 | 0 |
| front_loaded | 9 | 8 | 1 | 0 | 0 |
| saturating_imperfect_curve | 14 | 12 | 2 | 0 | 0 |
| steady_winner | 4 | 0 | 4 | 0 | 0 |
| tiny_sample_whale | 2 | 1 | 1 | 0 | 0 |
| tracking_outage | 3 | 1 | 2 | 0 | 0 |

Abstentions: recall 0.400 over 5 rows where insufficient_evidence is the only ideal action; 6 abstained, appropriateness 1.000; false abstentions 0; missed abstentions 3.

Disagreements with the committed expectations (forbidden or unlisted):

| scenario | channel | date | action | class | ideal | acceptable | forbidden | reason |
|---|---|---|---|---|---|---|---|---|
| delayed_reporting | unity | 2025-09-08 | decrease | forbidden | hold/insufficient_evidence | increase_test | decrease | upper bound of average D30 ROAS 0.470 is below the target 0.487 (policy heuristic on forecast average ROAS; not a guarantee of economic safety) |
| delayed_reporting | unity | 2025-10-06 | decrease | forbidden | hold/insufficient_evidence | increase_test | decrease | upper bound of average D30 ROAS 0.456 is below the target 0.496 (policy heuristic on forecast average ROAS; not a guarantee of economic safety) |
| delayed_reporting | unity | 2025-12-01 | decrease | forbidden | hold/insufficient_evidence | increase_test | decrease | upper bound of average D30 ROAS 0.281 is below the target 0.480 (policy heuristic on forecast average ROAS; not a guarantee of economic safety) |

Reasons behind every insufficient_evidence / hold (first stated reason, numbers masked):

| action | first stated reason (numbers masked) | channel-reviews |
|---|---|---|
| hold | average D# ROAS # is below the target #, so (diminishing-returns heuristic) there is no basis for an | 30 |
| hold | lower bound of model-estimated finite-step marginal D# ROAS # is below the target # (point estimate  | 18 |
| hold | quality_check_failed:cpi_anomaly | 1 |
| hold | an increase would be considered but is blocked: quality_check_failed:revenue_concentration | 1 |
| insufficient_evidence | forecast_covers_too_little_of_the_window | 5 |
| insufficient_evidence | quality_check_failed:tracking_gap | 2 |
| insufficient_evidence | quality_check_failed:window_min_installs | 1 |

#### Scenario validity: 17/17 generator calibration intents hold on this seed

No scenario was flagged on this seed.

<details><summary>every intent</summary>

| scenario | metric | value | op | threshold | pass |
|---|---|---|---|---|---|
| cpi_spike | cpi_spike_ratio_to_trailing_median | 2.099 | >= | 1.600 | True |
| delayed_reporting | min_report_lag_days | 6.000 | >= | 6.000 | True |
| delayed_reporting | other_channels_max_report_lag_days | 2.000 | <= | 2.000 | True |
| delayed_reporting | naive_visible_d3_revenue_fraction | 0.036 | <= | 0.600 | True |
| front_loaded | d3_net_revenue_per_install_rank | 1.000 | <= | 2.000 | True |
| front_loaded | d30_to_d3_multiplier_over_peer_median | 0.615 | <= | 0.650 | True |
| front_loaded | avg_d30_roas_over_target | 0.637 | <= | 0.900 | True |
| regime_shift | iap_per_active_day_post_over_pre | 0.732 | <= | 0.900 | True |
| regime_shift | ad_ecpm_post_over_pre | 1.105 | >= | 1.050 | True |
| saturating_imperfect_curve | avg_d30_roas_over_target | 1.808 | >= | 1.200 | True |
| saturating_imperfect_curve | true_step20_marginal_roas_over_target | 0.347 | <= | 0.800 | True |
| steady_winner | avg_d30_roas_over_target | 2.001 | >= | 1.400 | True |
| steady_winner | true_step20_marginal_roas_over_target | 1.538 | >= | 1.200 | True |
| tiny_sample_whale | window_installs_per_day_max | 8.000 | <= | 15.000 | True |
| tiny_sample_whale | whale_cohort_d3_net_over_normal_median | 47.978 | >= | 5.000 | True |
| tracking_outage | outage_event_drop_fraction | 0.928 | >= | 0.700 | True |
| tracking_outage | outage_install_change_abs | 0.073 | <= | 0.250 | True |

</details>

### Seed 9102

Artifact hashes (sha256, from provenance.json): decision_db `67582bf0b0b2`, hidden_truth `86c581481c1c`, decision_db_at_backtest_start `67582bf0b0b2`, forecast_store `f4c5a4477d33`

#### Forecasting (scored against D30 outcomes in storage; nominal interval coverage 0.80)

| level | group | n | wape | bias | n_with_interval | coverage | interval_width_over_forecast_mean | interval_width_over_forecast_median |
|---|---|---|---|---|---|---|---|---|
| cohort | overall | 1343 | 0.253 | -0.011 | 1024 | 0.792 | 0.754 | 0.790 |
| weekly_summary | overall | 214 | 0.136 | -0.011 | 139 | 0.748 | 0.385 | 0.405 |
| cohort | channel applovin | 282 | 0.281 | -0.002 | 218 | 0.780 | 0.814 | 0.805 |
| cohort | channel google_uac | 264 | 0.256 | -0.014 | 203 | 0.813 | 0.700 | 0.650 |
| cohort | channel meta | 281 | 0.245 | -0.030 | 218 | 0.784 | 0.788 | 0.779 |
| cohort | channel tiktok | 255 | 0.146 | 0.023 | 194 | 0.789 | 0.415 | 0.421 |
| cohort | channel unity | 261 | 0.301 | -0.018 | 191 | 0.796 | 1.045 | 1.017 |
| weekly_summary | channel applovin | 43 | 0.149 | -0.002 | 28 | 0.679 | 0.410 | 0.410 |
| weekly_summary | channel google_uac | 43 | 0.116 | -0.014 | 28 | 0.786 | 0.412 | 0.403 |
| weekly_summary | channel meta | 43 | 0.141 | -0.030 | 28 | 0.643 | 0.402 | 0.403 |
| weekly_summary | channel tiktok | 42 | 0.097 | 0.023 | 27 | 0.815 | 0.179 | 0.160 |
| weekly_summary | channel unity | 43 | 0.136 | -0.018 | 28 | 0.821 | 0.517 | 0.518 |

Accounting (cohorts): 1343 issued, 1343 scored, 0 pending, 310 withheld (below_min_installs: 24, insufficient_training_history: 140, low_activity: 10, revenue_spike: 136); share of matured D30 revenue withheld 0.318.

#### Policy and abstentions - PRIMARY (all scored rows, as registered)

48 scored channel-reviews: ideal 37, acceptable 5, forbidden 6, unlisted 0 (ideal 0.771, acceptable-or-better 0.875, forbidden 0.125).

| scenario | n_scored | ideal | acceptable | forbidden | unlisted |
|---|---|---|---|---|---|
| cpi_spike | 2 | 0 | 0 | 2 | 0 |
| delayed_reporting | 14 | 11 | 0 | 3 | 0 |
| front_loaded | 9 | 9 | 0 | 0 | 0 |
| saturating_imperfect_curve | 14 | 14 | 0 | 0 | 0 |
| steady_winner | 4 | 0 | 4 | 0 | 0 |
| tiny_sample_whale | 2 | 1 | 0 | 1 | 0 |
| tracking_outage | 3 | 2 | 1 | 0 | 0 |

Abstentions: recall 0.600 over 5 rows where insufficient_evidence is the only ideal action; 4 abstained, appropriateness 1.000; false abstentions 0; missed abstentions 6.

Disagreements with the committed expectations (forbidden or unlisted):

| scenario | channel | date | action | class | ideal | acceptable | forbidden | reason |
|---|---|---|---|---|---|---|---|---|
| cpi_spike | google_uac | 2025-09-22 | decrease | forbidden | hold | insufficient_evidence | increase_test/decrease | upper bound of average D30 ROAS 0.280 is below the target 0.534 (policy heuristic on forecast average ROAS; not a guarantee of economic safety) |
| cpi_spike | google_uac | 2025-09-29 | decrease | forbidden | hold | insufficient_evidence | increase_test/decrease | upper bound of average D30 ROAS 0.343 is below the target 0.524 (policy heuristic on forecast average ROAS; not a guarantee of economic safety) |
| delayed_reporting | unity | 2025-09-29 | decrease | forbidden | hold/insufficient_evidence | increase_test | decrease | upper bound of average D30 ROAS 0.463 is below the target 0.476 (policy heuristic on forecast average ROAS; not a guarantee of economic safety) |
| delayed_reporting | unity | 2025-12-01 | decrease | forbidden | hold/insufficient_evidence | increase_test | decrease | upper bound of average D30 ROAS 0.339 is below the target 0.507 (policy heuristic on forecast average ROAS; not a guarantee of economic safety) |
| delayed_reporting | unity | 2025-12-08 | decrease | forbidden | hold/insufficient_evidence | increase_test | decrease | upper bound of average D30 ROAS 0.387 is below the target 0.474 (policy heuristic on forecast average ROAS; not a guarantee of economic safety) |
| tiny_sample_whale | tiktok | 2025-12-01 | decrease | forbidden | insufficient_evidence | hold | increase_test/decrease | upper bound of average D30 ROAS 0.290 is below the target 0.881 (policy heuristic on forecast average ROAS; not a guarantee of economic safety) |

Reasons behind every insufficient_evidence / hold (first stated reason, numbers masked):

| action | first stated reason (numbers masked) | channel-reviews |
|---|---|---|
| hold | lower bound of model-estimated finite-step marginal D# ROAS # is below the target # (point estimate  | 26 |
| hold | average D# ROAS # is below the target #, so (diminishing-returns heuristic) there is no basis for an | 19 |
| hold | an increase would be considered but is blocked: quality_check_failed:revenue_concentration | 4 |
| insufficient_evidence | forecast_covers_too_little_of_the_window | 5 |
| insufficient_evidence | quality_check_failed:window_min_installs | 2 |
| insufficient_evidence | quality_check_failed:tracking_gap | 1 |

#### Scenario validity: 17/17 generator calibration intents hold on this seed

No scenario was flagged on this seed.

<details><summary>every intent</summary>

| scenario | metric | value | op | threshold | pass |
|---|---|---|---|---|---|
| cpi_spike | cpi_spike_ratio_to_trailing_median | 2.168 | >= | 1.600 | True |
| delayed_reporting | min_report_lag_days | 6.000 | >= | 6.000 | True |
| delayed_reporting | other_channels_max_report_lag_days | 2.000 | <= | 2.000 | True |
| delayed_reporting | naive_visible_d3_revenue_fraction | 0.049 | <= | 0.600 | True |
| front_loaded | d3_net_revenue_per_install_rank | 1.000 | <= | 2.000 | True |
| front_loaded | d30_to_d3_multiplier_over_peer_median | 0.630 | <= | 0.650 | True |
| front_loaded | avg_d30_roas_over_target | 0.661 | <= | 0.900 | True |
| regime_shift | iap_per_active_day_post_over_pre | 0.816 | <= | 0.900 | True |
| regime_shift | ad_ecpm_post_over_pre | 1.101 | >= | 1.050 | True |
| saturating_imperfect_curve | avg_d30_roas_over_target | 1.670 | >= | 1.200 | True |
| saturating_imperfect_curve | true_step20_marginal_roas_over_target | 0.332 | <= | 0.800 | True |
| steady_winner | avg_d30_roas_over_target | 1.882 | >= | 1.400 | True |
| steady_winner | true_step20_marginal_roas_over_target | 1.452 | >= | 1.200 | True |
| tiny_sample_whale | window_installs_per_day_max | 15.000 | <= | 15.000 | True |
| tiny_sample_whale | whale_cohort_d3_net_over_normal_median | 47.646 | >= | 5.000 | True |
| tracking_outage | outage_event_drop_fraction | 0.924 | >= | 0.700 | True |
| tracking_outage | outage_install_change_abs | 0.142 | <= | 0.250 | True |

</details>

### Seed 9103

Artifact hashes (sha256, from provenance.json): decision_db `70b3d5e0213d`, hidden_truth `3dca821f16cc`, decision_db_at_backtest_start `70b3d5e0213d`, forecast_store `f7f1aaf8c59d`

#### Forecasting (scored against D30 outcomes in storage; nominal interval coverage 0.80)

| level | group | n | wape | bias | n_with_interval | coverage | interval_width_over_forecast_mean | interval_width_over_forecast_median |
|---|---|---|---|---|---|---|---|---|
| cohort | overall | 1345 | 0.260 | -0.010 | 1026 | 0.768 | 0.803 | 0.845 |
| weekly_summary | overall | 215 | 0.137 | -0.010 | 140 | 0.786 | 0.379 | 0.440 |
| cohort | channel applovin | 279 | 0.269 | -0.031 | 217 | 0.765 | 0.912 | 0.898 |
| cohort | channel google_uac | 268 | 0.275 | -0.006 | 207 | 0.797 | 0.805 | 0.806 |
| cohort | channel meta | 275 | 0.287 | -0.002 | 210 | 0.786 | 0.957 | 0.922 |
| cohort | channel tiktok | 258 | 0.141 | 0.036 | 194 | 0.742 | 0.378 | 0.392 |
| cohort | channel unity | 265 | 0.274 | -0.029 | 198 | 0.747 | 0.934 | 0.950 |
| weekly_summary | channel applovin | 43 | 0.139 | -0.031 | 28 | 0.786 | 0.448 | 0.460 |
| weekly_summary | channel google_uac | 43 | 0.138 | -0.006 | 28 | 0.750 | 0.283 | 0.285 |
| weekly_summary | channel meta | 43 | 0.156 | -0.002 | 28 | 0.714 | 0.447 | 0.452 |
| weekly_summary | channel tiktok | 43 | 0.084 | 0.036 | 28 | 0.857 | 0.172 | 0.166 |
| weekly_summary | channel unity | 43 | 0.132 | -0.029 | 28 | 0.821 | 0.544 | 0.543 |

Accounting (cohorts): 1345 issued, 1345 scored, 0 pending, 308 withheld (below_min_installs: 23, insufficient_training_history: 143, low_activity: 7, revenue_spike: 135); share of matured D30 revenue withheld 0.344.

#### Policy and abstentions - PRIMARY (all scored rows, as registered)

48 scored channel-reviews: ideal 36, acceptable 5, forbidden 7, unlisted 0 (ideal 0.750, acceptable-or-better 0.854, forbidden 0.146).

| scenario | n_scored | ideal | acceptable | forbidden | unlisted |
|---|---|---|---|---|---|
| cpi_spike | 2 | 1 | 0 | 1 | 0 |
| delayed_reporting | 14 | 10 | 0 | 4 | 0 |
| front_loaded | 9 | 8 | 1 | 0 | 0 |
| saturating_imperfect_curve | 14 | 14 | 0 | 0 | 0 |
| steady_winner | 4 | 1 | 3 | 0 | 0 |
| tiny_sample_whale | 2 | 1 | 0 | 1 | 0 |
| tracking_outage | 3 | 1 | 1 | 1 | 0 |

Abstentions: recall 0.400 over 5 rows where insufficient_evidence is the only ideal action; 3 abstained, appropriateness 1.000; false abstentions 0; missed abstentions 7.

Disagreements with the committed expectations (forbidden or unlisted):

| scenario | channel | date | action | class | ideal | acceptable | forbidden | reason |
|---|---|---|---|---|---|---|---|---|
| cpi_spike | google_uac | 2025-09-22 | decrease | forbidden | hold | insufficient_evidence | increase_test/decrease | upper bound of average D30 ROAS 0.307 is below the target 0.547 (policy heuristic on forecast average ROAS; not a guarantee of economic safety) |
| delayed_reporting | unity | 2025-09-29 | decrease | forbidden | hold/insufficient_evidence | increase_test | decrease | upper bound of average D30 ROAS 0.353 is below the target 0.454 (policy heuristic on forecast average ROAS; not a guarantee of economic safety) |
| delayed_reporting | unity | 2025-11-17 | decrease | forbidden | hold/insufficient_evidence | increase_test | decrease | upper bound of average D30 ROAS 0.325 is below the target 0.517 (policy heuristic on forecast average ROAS; not a guarantee of economic safety) |
| delayed_reporting | unity | 2025-11-24 | decrease | forbidden | hold/insufficient_evidence | increase_test | decrease | upper bound of average D30 ROAS 0.372 is below the target 0.521 (policy heuristic on forecast average ROAS; not a guarantee of economic safety) |
| delayed_reporting | unity | 2025-12-01 | decrease | forbidden | hold/insufficient_evidence | increase_test | decrease | upper bound of average D30 ROAS 0.344 is below the target 0.525 (policy heuristic on forecast average ROAS; not a guarantee of economic safety) |
| tiny_sample_whale | tiktok | 2025-12-01 | decrease | forbidden | insufficient_evidence | hold | increase_test/decrease | upper bound of average D30 ROAS 0.262 is below the target 0.873 (policy heuristic on forecast average ROAS; not a guarantee of economic safety) |
| tracking_outage | google_uac | 2025-11-24 | decrease | forbidden | insufficient_evidence | hold | decrease/increase_test | upper bound of average D30 ROAS 0.292 is below the target 0.523 (policy heuristic on forecast average ROAS; not a guarantee of economic safety) |

Reasons behind every insufficient_evidence / hold (first stated reason, numbers masked):

| action | first stated reason (numbers masked) | channel-reviews |
|---|---|---|
| hold | lower bound of model-estimated finite-step marginal D# ROAS # is below the target # (point estimate  | 24 |
| hold | average D# ROAS # is below the target #, so (diminishing-returns heuristic) there is no basis for an | 20 |
| hold | an increase would be considered but is blocked: quality_check_failed:revenue_concentration | 3 |
| insufficient_evidence | quality_check_failed:tracking_gap | 2 |
| insufficient_evidence | forecast_covers_too_little_of_the_window | 1 |
| insufficient_evidence | quality_check_failed:window_min_installs | 1 |

#### Scenario validity: 17/17 generator calibration intents hold on this seed

No scenario was flagged on this seed.

<details><summary>every intent</summary>

| scenario | metric | value | op | threshold | pass |
|---|---|---|---|---|---|
| cpi_spike | cpi_spike_ratio_to_trailing_median | 1.764 | >= | 1.600 | True |
| delayed_reporting | min_report_lag_days | 6.000 | >= | 6.000 | True |
| delayed_reporting | other_channels_max_report_lag_days | 2.000 | <= | 2.000 | True |
| delayed_reporting | naive_visible_d3_revenue_fraction | 0.200 | <= | 0.600 | True |
| front_loaded | d3_net_revenue_per_install_rank | 1.000 | <= | 2.000 | True |
| front_loaded | d30_to_d3_multiplier_over_peer_median | 0.611 | <= | 0.650 | True |
| front_loaded | avg_d30_roas_over_target | 0.672 | <= | 0.900 | True |
| regime_shift | iap_per_active_day_post_over_pre | 0.572 | <= | 0.900 | True |
| regime_shift | ad_ecpm_post_over_pre | 1.103 | >= | 1.050 | True |
| saturating_imperfect_curve | avg_d30_roas_over_target | 1.751 | >= | 1.200 | True |
| saturating_imperfect_curve | true_step20_marginal_roas_over_target | 0.314 | <= | 0.800 | True |
| steady_winner | avg_d30_roas_over_target | 2.053 | >= | 1.400 | True |
| steady_winner | true_step20_marginal_roas_over_target | 1.605 | >= | 1.200 | True |
| tiny_sample_whale | window_installs_per_day_max | 4.000 | <= | 15.000 | True |
| tiny_sample_whale | whale_cohort_d3_net_over_normal_median | 46.393 | >= | 5.000 | True |
| tracking_outage | outage_event_drop_fraction | 0.928 | >= | 0.700 | True |
| tracking_outage | outage_install_change_abs | 0.183 | <= | 0.250 | True |

</details>

## Pooled counts (sums of the per-seed counts above; not a replacement for them)

| seeds | n_scored | ideal | acceptable | forbidden | unlisted | forbidden_rate | acceptable_or_better_rate |
|---|---|---|---|---|---|---|---|
| 9101, 9102, 9103 | 144 | 107 | 21 | 16 | 0 | 0.111 | 0.889 |

Forecast metrics are not pooled (a ratio of sums cannot be rebuilt from per-seed ratios); per-seed cohort-level overall values:

| seed | n | wape | bias | n_with_interval | coverage | interval_width_over_forecast_mean | interval_width_over_forecast_median |
|---|---|---|---|---|---|---|---|
| 9101.000 | 1329.000 | 0.267 | -0.031 | 1004.000 | 0.783 | 0.796 | 0.824 |
| 9102.000 | 1343.000 | 0.253 | -0.011 | 1024.000 | 0.792 | 0.754 | 0.790 |
| 9103.000 | 1345.000 | 0.260 | -0.010 | 1026.000 | 0.768 | 0.803 | 0.845 |

## Limitations

- Three synthetic seeds from the generator the rules were developed against: this is not evidence about real data or other generators.
- Interval coverage is compared with nominal 0.80; any shortfall above is a finding, not something corrected.
- A `forbidden` row is a violation of a committed scenario expectation, not independently proven economic harm.
- Only the deterministic baseline was run. The constraints suite and experiment safeguards are seed-independent and are reported in the development evaluation.

