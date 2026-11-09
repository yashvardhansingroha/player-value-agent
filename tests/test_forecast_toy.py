"""A toy end-to-end forecast small enough to verify by hand, run through the real review pipeline.

Channel 'meta', 2 players per cohort, revenue = ad revenue only (so net = ad). Cohort install days and revenues:

  cohort  install day  D0-D3 net (y3)  D0-D30 net (y30)  notes
  c0      0            10              30                  D30 already complete at the review (day 34)
  c1      1            20              50                  same
  c2      2            10              40                  same
  c3      29           15              60                  newly decision-ready at day 34 -> gets a forecast
  c4      28            0               0                  zero activity at all: zero-D3 fallback

Training = c0..c2: sum(y3) = 40, sum(y30) = 120, multiplier = 120 / 40 = 3.0 (mean of the three ratios would be 3.1667).
Forecast c3 = 3.0 x 15 = 45. Reference (ignores D3) = value per install x installs = (120 / 6) x 2 = 40.
Zero-D3 cohort c4: forecast = reference = 20 x 2 = 40 (NOT 3.0 x 0 = 0).
Scored at day 70 (D30 outcomes of c3, c4 complete): sum(actual) = 60, sum(forecast) = 85,
WAPE = (|60-45| + |0-40|) / 60 = 55/60, bias = (85-60)/60.
"""
import json
import sqlite3

import pandas as pd
import pytest

from src.data.calendar import day_to_date
from src.data.load import create_database
from src.forecast import scoring, store
from src.forecast.pipeline import START, load_config, run_review

REVIEW_DAY, SCORING_DAY = 34, 70
D = lambda n: day_to_date(START, n)
ROW = dict(sessions=2, levels=2, ad_impressions=0, iap_gross=0.0)

# (cohort id, install day, y3, extra revenue between D4 and D30 so that y30 = y3 + extra, active?)
COHORTS = [("c0", 0, 10, 20), ("c1", 1, 20, 30), ("c2", 2, 10, 30), ("c3", 29, 15, 45), ("c4", 28, 0, 0)]


def build_toy(path):
    players, days, spend, reporting = [], [], [], []
    pid = 0
    for name, day, y3, extra in COHORTS:
        ids = [pid + 1, pid + 2]
        pid += 2
        for i in ids:
            players.append(dict(player_id=i, channel="meta", install_date=D(day), reported_at=D(day + 1)))
        if name != "c4":  # c4 has no activity rows at all
            half = y3 / 2
            for i in ids:
                days.append(dict(player_id=i, event_date=D(day + 1), day_n=1, ad_revenue=half, reported_at=D(day + 2), **ROW))
                days.append(dict(player_id=i, event_date=D(day + 10), day_n=10, ad_revenue=extra / 2, reported_at=D(day + 11), **ROW))
        spend.append(dict(channel="meta", spend_date=D(day), spend=10.0, reported_at=D(day + 1)))
        reporting.append(dict(channel="meta", install_date=D(day), through_day=3, complete_at=D(day + 4)))
        reporting.append(dict(channel="meta", install_date=D(day), through_day=30, complete_at=D(day + 31)))
    create_database(path, pd.DataFrame(players), pd.DataFrame(days), pd.DataFrame(spend), pd.DataFrame(reporting),
                    {"store_fee": "0.3", "start_date": START})


def toy_cfg():
    cfg = load_config()
    cfg["quality"].update(min_installs_per_cohort=1, baseline_min_cohorts=50, window_min_installs=1)
    cfg["forecast"].update(min_train_cohorts=3, min_pooled_cohorts=3, cohort_min_calibration=30, summary_min_calibration=12)
    return cfg


@pytest.fixture()
def toy(tmp_path):
    build_toy(tmp_path / "toy.db")
    dec = sqlite3.connect(tmp_path / "toy.db")
    st = store.create_store(tmp_path / "toy_forecasts.db")
    log = run_review(dec, st, D(REVIEW_DAY), toy_cfg())
    yield dec, st, log
    dec.close()
    st.close()


def issued(st):
    return pd.read_sql_query("SELECT * FROM forecast_issued", st).set_index("install_date")


def test_toy_training_multiplier_and_forecast_by_hand(toy):
    _, st, log = toy
    f = issued(st)
    c3 = f.loc[D(29)]
    assert c3.method == "channel_multiplier" and c3.train_n == 3
    training = json.loads(c3.training_json)
    assert training["sum_y3"] == pytest.approx(40.0) and training["sum_y30"] == pytest.approx(120.0)
    assert training["multiplier"] == pytest.approx(3.0)  # ratio of sums, not the 3.1667 mean of ratios
    assert c3.point_forecast == pytest.approx(45.0)  # 3.0 x 15
    assert c3.observed_d3_floor == pytest.approx(15.0) and c3.point_floor_adjusted == 0  # 45 is already above the 15 already observed
    assert c3.reference_forecast == pytest.approx(40.0)  # (120 / 6 installs) x 2 installs
    assert c3.train_max_outcome_complete_at == D(2 + 31)  # latest training outcome became complete on day 33 <= review day 34


def test_toy_cohorts_with_known_outcome_are_not_forecast_and_the_reason_is_stored(toy):
    _, st, log = toy
    held = pd.read_sql_query("SELECT * FROM forecast_withheld", st).set_index("install_date")
    assert set(held.index) == {D(0), D(1), D(2)}
    assert (held.reason == "outcome_already_available").all()
    assert log["issued"] == 2 and log["withheld"] == {"outcome_already_available": 3}


def test_toy_zero_d3_uses_fallback_not_zero_and_zero_activity_retention_is_zero_not_missing(toy):
    _, st, _ = toy
    c4 = issued(st).loc[D(28)]
    assert c4.method == "zero_d3_fallback"
    assert c4.point_forecast == pytest.approx(40.0)  # 20 per install x 2, not 3.0 x 0
    assert c4.interval_status == "unavailable" and "non_standard_method" in c4.interval_reason
    features = json.loads(c4.features_json)
    assert features["d0_3_net_revenue"] == 0.0 and features["zero_d3"] is True
    assert features["d1_retention"] == 0.0  # zero activity is 0, distinct from NULL (unavailable)


def test_toy_interval_is_explicitly_unavailable_without_calibration_history(toy):
    _, st, _ = toy
    c3 = issued(st).loc[D(29)]
    assert c3.interval_status == "unavailable" and c3.interval_lo is None or pd.isna(c3.interval_lo)
    assert "insufficient_calibration_history" in c3.interval_reason and c3.calib_n == 0


def test_toy_weekly_summary_lists_included_cohorts_and_is_separate_from_cohort_forecasts(toy):
    _, st, _ = toy
    s = pd.read_sql_query("SELECT * FROM weekly_summary", st).iloc[0]
    assert (s.window_start, s.window_end) == (D(23), D(29))  # 7 install dates ending at the latest decision-ready date
    assert json.loads(s.included_json) == [D(28), D(29)]
    assert s.n_included == 2 and s.installs_included == 4 and s.spend_included == pytest.approx(20.0)
    assert s.point_forecast == pytest.approx(85.0)  # 45 + 40
    quality = {c["name"]: c for c in json.loads(s.quality_json)}
    assert quality["missing_days"]["passed"] is False  # days 23..27 have no cohort data: reported, not silently ignored
    assert len(quality["missing_days"]["evidence"]["missing_dates"]) == 5
    # every expected date is accounted for; missing dates make the summary not fully decision-ready and give it no interval
    accounting = {a["install_date"]: a["status"] for a in json.loads(s.accounting_json)}
    assert accounting == {D(23): "missing", D(24): "missing", D(25): "missing", D(26): "missing", D(27): "missing", D(28): "forecast", D(29): "forecast"}
    assert s.n_expected_dates == 7 and s.fully_decision_ready == 0 and s.interval_status == "unavailable"
    assert s.interval_reason.startswith("window_not_fully_decision_ready")
    assert s.observed_d3_floor == pytest.approx(15.0) and s.point_forecast >= s.observed_d3_floor  # floor = 15 + 0


def test_toy_scoring_scores_only_matured_outcomes_and_matches_hand_numbers(toy):
    dec, st, _ = toy
    early = scoring.score_cohorts(st, dec, D(REVIEW_DAY))  # c3, c4 D30 outcomes are not complete yet
    assert early["scored"] == 0 and early["pending_outcome_not_in_storage"] == 2
    late = scoring.score_cohorts(st, dec, D(SCORING_DAY))
    o = late["overall"]
    assert late["scored"] == 2 and late["pending_outcome_not_in_storage"] == 0
    assert o["sum_actual"] == pytest.approx(60.0) and o["sum_forecast"] == pytest.approx(85.0)
    assert o["wape"] == pytest.approx(55 / 60) and o["bias"] == pytest.approx(25 / 60)
    assert late["withheld"]["outcome_already_available"]["cohorts"] == 3


def test_store_is_immutable(toy):
    _, st, _ = toy
    for sql in ("UPDATE forecast_issued SET point_forecast = 0", "DELETE FROM forecast_issued",
                "UPDATE forecast_withheld SET reason = 'x'", "DELETE FROM forecast_withheld",
                "UPDATE weekly_summary SET n_included = 0", "DELETE FROM weekly_summary"):
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            st.execute(sql)
