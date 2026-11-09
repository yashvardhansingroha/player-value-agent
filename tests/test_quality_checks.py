"""Quality checks: structured results with reasons, evaluable/not-evaluable, and the cohort annotation rules."""
import numpy as np
import pandas as pd
import pytest

from src.forecast.pipeline import load_config
from src.quality import checks

Q = load_config()["quality"]


def test_results_are_structured_and_json_ready():
    r = checks.check_min_installs(12, 30)
    d = r.to_dict()
    assert d["name"] == "min_installs" and d["passed"] is False and "below the minimum 30" in d["reason"]
    assert d["evidence"] == {"installs": 12, "minimum": 30} and d["evaluable"] is True


def test_zero_missing_and_decision_ready_checks():
    assert not checks.check_zero_d3_revenue(0.0).passed and checks.check_zero_d3_revenue(0.5).passed
    assert not checks.check_zero_d3_revenue(float("nan")).passed
    assert not checks.check_missing_spend(float("nan")).passed and checks.check_missing_spend(3.0).passed
    assert not checks.check_decision_ready(0).passed and checks.check_decision_ready(1).passed


def test_spike_and_low_activity_checks_and_not_evaluable_without_baseline():
    assert not checks.check_revenue_spike(4.0, 1.0, 3.0).passed and checks.check_revenue_spike(2.9, 1.0, 3.0).passed
    assert not checks.check_low_activity(0.10, 0.30, 0.4).passed and checks.check_low_activity(0.13, 0.30, 0.4).passed
    nb = checks.check_revenue_spike(4.0, float("nan"), 3.0)
    assert nb.passed and nb.evaluable is False and "not evaluable" in nb.reason  # never a silent pass of real evidence


def test_concentration_cpi_missing_days_and_window_size():
    flat = checks.check_revenue_concentration([1.0] * 100, 0.5)  # top 1 payer of 100 -> 1%
    assert flat.passed and flat.evidence["top_k"] == 1 and flat.evidence["top_share"] == pytest.approx(0.01)
    whale = checks.check_revenue_concentration([1000.0] + [1.0] * 99, 0.5)
    assert not whale.passed and whale.evidence["top_share"] == pytest.approx(1000 / 1099)
    assert checks.check_revenue_concentration([], 0.5).evaluable is False
    assert not checks.check_cpi_anomaly(2.0, 1.0, 1.5, 0.6).passed and checks.check_cpi_anomaly(1.2, 1.0, 1.5, 0.6).passed
    assert checks.check_cpi_anomaly(1.2, float("nan"), 1.5, 0.6).evaluable is False
    m = checks.check_missing_days(["d1", "d2", "d3"], ["d1", "d3"])
    assert not m.passed and m.evidence["missing_dates"] == ["d2"]
    assert not checks.check_window_min_installs(100, 150).passed
    assert checks.check_reporting_completeness([], ["a", "b", "c", "d"], "w", 2).passed  # ordinary lag: 4 <= 2 + 2 expected
    assert not checks.check_reporting_completeness([], list("abcde"), "w", 2).passed  # more pending than the lag explains


def daily(n_days, drop_last=0, drop_to=0.1):
    dates = pd.date_range("2025-03-01", periods=n_days).strftime("%Y-%m-%d")
    active = np.full(n_days, 100.0)
    if drop_last:
        active[-drop_last:] *= drop_to
    return pd.DataFrame({"event_date": dates, "active_player_days": active, "recent_installs": 500})


def test_tracking_gap_flags_a_collapse_but_not_reporting_delay():
    q = Q
    df = daily(40, drop_last=7, drop_to=0.1)  # last 7 days collapse to 10% of normal
    as_of = "2025-04-12"  # the day after the last row (2025-04-09 + lag 1 + 1)
    assert not checks.check_tracking_gap(df, 1, as_of, q).passed
    # the same collapse in rows that are within the declared lag of as_of is IGNORED (they may simply be unreported)
    late_only = daily(40)
    late_only.loc[late_only.index[-2:], "active_player_days"] = 0.0
    assert checks.check_tracking_gap(late_only, 3, "2025-04-11", q).passed
    assert checks.check_tracking_gap(daily(10), 1, "2025-03-12", q).evaluable is False


def test_tracking_gap_straddling_outage_is_flagged_and_short_dips_are_not():
    """A 7-day outage that straddles the recent and baseline windows. The first design compared the mean of the last 7 days
    with the mean of the 14 days before (ratio 0.51 here, which passed a 0.5 floor); the revised check uses a median
    baseline and consecutive low days."""
    df = daily(60)
    df.loc[50:56, "active_player_days"] = 5.0  # 7 low days; as_of below makes the last complete day index 59
    as_of = "2025-05-01"  # last row 2025-04-29 (index 59) + lag 1 + 1
    res = checks.check_tracking_gap(df, 1, as_of, Q)
    assert not res.passed and res.evidence["longest_run_days"] == 4
    assert res.evidence["baseline_median_per_player"] == pytest.approx(100 / 500)
    old_ratio = df.iloc[53:60].eval("active_player_days / recent_installs").mean() / df.iloc[39:53].eval("active_player_days / recent_installs").mean()
    assert old_ratio > 0.5  # the earlier mean-vs-mean design would have passed this outage
    short = daily(60)
    short.loc[55:56, "active_player_days"] = 5.0  # a 2-day dip is below the 3-day run limit
    assert checks.check_tracking_gap(short, 1, as_of, Q).passed


def test_window_cpi_uses_every_decision_ready_cohort_not_only_forecast_ones():
    base = pd.DataFrame({"install_date": pd.date_range("2025-03-01", periods=37).strftime("%Y-%m-%d"), "d3_complete": 1,
                         "spend": 100.0, "installs_reported": 100, "cpi": 1.0})
    window = base.iloc[30:]  # 7 cohorts: five with halved installs (CPI 2.0), two normal
    base.loc[base.index[30:35], "installs_reported"] = 50
    base.loc[base.index[30:35], "cpi"] = 2.0
    res = checks.check_window_cpi(base, list(window.install_date), Q)
    assert not res.passed and res.evidence["ratio"] == pytest.approx(700 / 450) and res.evidence["cohorts_in_window"] == 7
    ok = checks.check_window_cpi(cohort_frame(40).assign(cpi=1.0), pd.date_range("2025-03-31", periods=7).strftime("%Y-%m-%d").tolist(), Q)
    assert ok.passed


def cpi_frame(window_rows, prior_n=10):
    """prior_n normal cohorts (spend 100, installs 100, CPI 1.0) followed by window rows (spend, installs, d3_complete)."""
    prior = pd.DataFrame({"install_date": pd.date_range("2025-03-01", periods=prior_n).strftime("%Y-%m-%d"), "d3_complete": 1,
                          "spend": 100.0, "installs_reported": 100, "cpi": 1.0})
    dates = pd.date_range("2025-03-01", periods=prior_n + len(window_rows)).strftime("%Y-%m-%d")[prior_n:]
    win = pd.DataFrame([{"install_date": d, "d3_complete": c, "spend": s, "installs_reported": i, "cpi": (s / i if (i > 0 and c == 1) else np.nan)}
                        for d, (s, i, c) in zip(dates, window_rows)])
    return pd.concat([prior, win], ignore_index=True), list(dates)


def test_window_cpi_keeps_zero_install_spend_in_the_aggregate():
    df, dates = cpi_frame([(100.0, 100, 1), (100.0, 0, 1)])  # day 1 CPI 1; day 2 spent 100 and bought nothing
    res = checks.check_window_cpi(df, dates, Q)
    assert res.evidence["spend"] == 200.0 and res.evidence["installs"] == 100
    assert res.evidence["window_cpi"] == pytest.approx(2.0)  # 200 / 100, NOT 1.0 (the old filter dropped the zero-install day)
    assert res.evidence["ratio"] == pytest.approx(2.0) and not res.passed  # 2.0 x the trailing median exceeds the 1.5 limit


def test_window_cpi_wholly_zero_install_window_is_explicitly_unusable_not_normal():
    df, dates = cpi_frame([(100.0, 0, 1), (80.0, 0, 1)])
    res = checks.check_window_cpi(df, dates, Q)
    assert res.passed is False and res.evaluable is True and "zero installs" in res.reason and "unusable" in res.reason
    assert res.evidence["spend"] == 180.0 and res.evidence["installs"] == 0
    nothing, ndates = cpi_frame([(0.0, 0, 1)])  # no spend and no installs: simply nothing to evaluate
    assert checks.check_window_cpi(nothing, ndates, Q).evaluable is False


def test_window_cpi_unknown_installs_are_excluded_and_listed_not_counted_as_zero():
    df, dates = cpi_frame([(100.0, 100, 1), (100.0, 7, 0), (100.0, 100, 1)])  # middle cohort: D3 incomplete, 7 installs so far
    res = checks.check_window_cpi(df, dates, Q)
    assert res.evidence["spend"] == 200.0 and res.evidence["installs"] == 200  # incomplete cohort contributes neither spend nor installs
    assert res.evidence["excluded_incomplete_dates"] == [dates[1]] and res.evidence["window_cpi"] == pytest.approx(1.0) and res.passed
    # a KNOWN zero is different: it is included and moves the CPI
    known_zero, kdates = cpi_frame([(100.0, 100, 1), (100.0, 0, 1), (100.0, 100, 1)])
    assert checks.check_window_cpi(known_zero, kdates, Q).evidence["window_cpi"] == pytest.approx(300 / 200)


def test_reporting_completeness_separates_interior_gaps_from_ordinary_lag():
    ordinary = checks.check_reporting_completeness([], ["d1", "d2", "d3", "d4"], "w", 2)
    assert ordinary.passed and ordinary.evidence["interior_incomplete_dates"] == []
    interior = checks.check_reporting_completeness(["2025-08-31"], ["d1", "d2"], "w", 2)
    assert not interior.passed and interior.evidence["interior_incomplete_dates"] == ["2025-08-31"]
    assert "INSIDE the window" in interior.reason and "partial data, not zero" in interior.reason


def test_concentration_reports_its_eligible_population_and_is_unavailable_without_one():
    window = ["d1", "d2", "d3"]
    partial = checks.check_revenue_concentration([1000.0] + [1.0] * 99, 0.5, ["d1", "d3"], window)
    assert not partial.passed and partial.evidence["excluded_dates"] == ["d2"] and "2 of 3 window dates" in partial.reason
    none = checks.check_revenue_concentration([], 0.5, [], window)
    assert none.evaluable is False and "no decision-ready cohort" in none.reason


def cohort_frame(n=20, **over):
    dates = pd.date_range("2025-03-01", periods=n).strftime("%Y-%m-%d")
    df = pd.DataFrame({
        "channel": "meta", "install_date": dates, "installs_reported": 80, "spend": 80.0, "d3_complete": 1,
        "cpi": 1.0, "d1_retention": 0.30, "d3_retention": 0.2, "d0_3_net_revenue": 40.0, "d0_3_gross_iap": 10.0,
        "d0_3_ad_revenue": 33.0, "payer_rate_d0_3": 0.03, "arpdau_d0_3": 0.1, "sessions_per_active_user": 2.0, "levels_per_active_user": 3.0})
    for k, (i, v) in over.items():
        df.loc[i, k] = v
    return df


def test_annotate_flags_each_exclusion_reason_and_uses_only_prior_cohorts():
    df = cohort_frame()
    df.loc[15, "d0_3_net_revenue"] = 40.0 * 80 * 4  # spike: 4x the median revenue per install
    df.loc[16, "d1_retention"] = 0.05  # low activity
    df.loc[17, "installs_reported"] = 10  # too small
    df.loc[18, "spend"] = np.nan  # missing spend
    df.loc[19, "d0_3_net_revenue"] = 0.0  # zero D3 revenue
    df.loc[3, "d3_complete"] = 0  # not decision-ready: no reason assigned, simply not ready
    for c in ("cpi", "d1_retention", "d0_3_net_revenue"):
        df.loc[3, c] = np.nan
    ann = checks.annotate_cohorts(df, Q).set_index("install_date")
    reason = ann.exclusion_reason
    assert reason["2025-03-16"] == "revenue_spike" and reason["2025-03-17"] == "low_activity"
    assert reason["2025-03-18"] == "below_min_installs" and reason["2025-03-19"] == "missing_spend"
    assert pd.isna(reason["2025-03-20"]) and bool(ann.zero_d3["2025-03-20"])  # zero D3 is a method flag, not an exclusion
    assert pd.isna(reason["2025-03-04"])  # not decision-ready
    # early cohorts: fewer than baseline_min_cohorts priors, so spike/low-activity are not evaluable -> not excluded
    assert reason.iloc[:10].isna().all()


def test_spike_in_the_first_cohorts_cannot_set_its_own_baseline():
    df = cohort_frame(12)
    df.loc[0, "d0_3_net_revenue"] = 1e6  # huge, but with no prior cohorts it has no baseline to be compared with
    ann = checks.annotate_cohorts(df, Q)
    assert ann.exclusion_reason.isna().all()  # not evaluable (needs 10 prior cohorts), and shift(1) excludes itself
    assert ann.base_d3_net_per_install.iloc[:10].isna().all()
