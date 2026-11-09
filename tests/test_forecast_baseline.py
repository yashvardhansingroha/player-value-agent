"""Pure-function tests for the baseline multiplier, pooling/fallback rules and intervals (hand-checkable numbers)."""
import numpy as np
import pandas as pd
import pytest

from src.forecast import baseline, intervals


def cohorts(n, y3=10.0, y30=30.0, installs=50, start=0):
    return pd.DataFrame({"install_date": [f"2025-01-{i + 1:02d}" for i in range(start, start + n)] if n <= 28 else [f"d{i:04d}" for i in range(start, start + n)],
                         "y3": y3, "y30": y30, "installs": installs})


def test_multiplier_is_ratio_of_sums_not_mean_of_ratios():
    train = pd.DataFrame({"install_date": ["a", "b", "c"], "y3": [10.0, 20.0, 10.0], "y30": [30.0, 50.0, 40.0], "installs": [100, 200, 100]})
    info = baseline.fit_multiplier(train)
    assert info["multiplier"] == pytest.approx(120 / 40)  # 3.0; the mean of 3, 2.5, 4 would be 3.1667
    assert info["value_per_install"] == pytest.approx(120 / 400)
    assert info["n_cohorts"] == 3


def test_forecast_and_reference_and_zero_d3_fallback():
    info = {"multiplier": 3.0, "value_per_install": 0.3}
    normal = baseline.forecast_cohort(15.0, 50, "channel", info)
    assert normal["method"] == "channel_multiplier" and normal["point"] == pytest.approx(45.0) and normal["reference"] == pytest.approx(15.0)
    pooled = baseline.forecast_cohort(15.0, 50, "pooled", info)
    assert pooled["method"] == "pooled_multiplier"
    for zero in (0.0, None, float("nan")):
        z = baseline.forecast_cohort(zero, 50, "channel", info)
        assert z["method"] == "zero_d3_fallback" and z["point"] == pytest.approx(15.0)  # never 3.0 x 0 = 0


def test_no_multiplier_when_training_revenue_is_zero_or_empty():
    assert baseline.fit_multiplier(cohorts(5, y3=0.0, y30=0.0)) is None
    assert baseline.fit_multiplier(cohorts(0)) is None


CFG = {"forecast": {"train_window_cohorts": 60, "min_train_cohorts": 20, "min_pooled_cohorts": 40}}


def test_channel_level_pooled_level_and_insufficient_history_rules():
    big, small, tiny = cohorts(25), cohorts(10, y30=60.0), cohorts(3)
    level, info, _ = baseline.choose_training("a", {"a": big, "b": small}, CFG)
    assert level == "channel" and info["multiplier"] == pytest.approx(3.0)
    # channel c has 10 cohorts (< 20) -> pooled over all channels (25 + 10 + 10 = 45 >= 40), multiplier from all of them
    level, info, _ = baseline.choose_training("c", {"a": big, "b": small, "c": small}, CFG)
    assert level == "pooled" and info["n_cohorts"] == 45
    assert info["multiplier"] == pytest.approx((25 * 30 + 20 * 60) / (45 * 10))
    # not enough anywhere -> no training, with the counts reported
    level, info, why = baseline.choose_training("c", {"a": small, "c": tiny}, CFG)
    assert level is None and info is None and why["channel_training_cohorts"] == 3 and why["pooled_training_cohorts"] == 13


def test_training_uses_only_the_most_recent_window():
    old, recent = cohorts(40, y30=300.0, start=0), cohorts(60, y30=30.0, start=40)  # install_date sorts lexicographically
    both = pd.concat([old, recent], ignore_index=True)
    info = baseline.fit_multiplier(baseline.select_training(both, 60))
    assert info["n_cohorts"] == 60 and info["multiplier"] == pytest.approx(3.0)  # the 40 old cohorts are outside the window


def test_interval_uses_empirical_quantiles_and_is_unavailable_without_enough_history():
    ratios = list(np.linspace(0.5, 1.5, 101))  # quantile 0.1 = 0.6, quantile 0.9 = 1.4
    iv = intervals.interval_from_ratios(ratios, 100.0, 0.20, 30)
    assert iv["status"] == "available" and iv["calib_n"] == 101
    assert iv["lo"] == pytest.approx(60.0) and iv["hi"] == pytest.approx(140.0)
    short = intervals.interval_from_ratios(ratios[:10], 100.0, 0.20, 30)
    assert short["status"] == "unavailable" and short["lo"] is None and short["hi"] is None
    assert "insufficient_calibration_history" in short["reason"] and short["calib_n"] == 10


def test_floor_raises_lo_and_keeps_bounds_ordered_without_widening():
    ratios = list(np.linspace(0.5, 1.5, 101))
    iv = intervals.interval_from_ratios(ratios, 100.0, 0.20, 30)  # [60, 140]
    lo, hi, adj = intervals.apply_floor(iv["lo"], iv["hi"], 90.0)  # observed D3 revenue is 90: D30 cannot be below it
    assert (lo, hi, adj) == (pytest.approx(90.0), pytest.approx(140.0), True)
    assert lo >= iv["lo"] and hi >= iv["hi"] and (hi - lo) < (iv["hi"] - iv["lo"])  # narrower, never wider
    # entire interval below the floor: both bounds rise to the floor and stay ordered (lo <= hi)
    low = intervals.interval_from_ratios(list(np.linspace(0.1, 0.2, 101)), 100.0, 0.20, 30)
    lo2, hi2, adj2 = intervals.apply_floor(low["lo"], low["hi"], 90.0)
    assert lo2 == pytest.approx(90.0) and hi2 == pytest.approx(90.0) and lo2 <= hi2 and adj2
    # floor below the lower bound: nothing changes
    assert intervals.apply_floor(60.0, 140.0, 10.0) == (60.0, 140.0, False)
    # unavailable intervals pass through untouched
    assert intervals.apply_floor(None, None, 90.0) == (None, None, False)


def test_multiplier_forecast_is_at_least_observed_d3_when_training_is_consistent():
    train = pd.DataFrame({"install_date": ["a", "b"], "y3": [10.0, 20.0], "y30": [10.0, 25.0], "installs": [1, 1]})  # y30 >= y3
    info = baseline.fit_multiplier(train)
    assert info["multiplier"] >= 1.0 and baseline.forecast_cohort(40.0, 5, "channel", info)["point"] >= 40.0


def test_matured_ratios_only_use_forecasts_whose_outcomes_are_supplied():
    earlier = pd.DataFrame({"channel": "m", "install_date": ["a", "b", "c"], "point": [10.0, 20.0, 0.0]})
    outcomes = pd.DataFrame({"channel": "m", "install_date": ["a", "b"], "y30": [15.0, 10.0]})  # 'c' has no matured outcome
    done = pd.DataFrame({"channel": "m", "install_date": ["a", "b"], "complete30": ["2025-02-01", "2025-02-03"]})
    ratios, last = intervals.matured_ratios(earlier, outcomes, done, 100)
    assert ratios == [1.5, 0.5] and last == "2025-02-03"
