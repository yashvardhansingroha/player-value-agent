"""Backtest pipeline on a smaller generated dataset: leakage, issuance-once, calibration timing, explicit limitations,
and historical invariance of the forecasts themselves. Selected tests; they do not prove every leakage path is closed."""
import json
import shutil
import sqlite3

import pandas as pd
import pytest

from src.data.calendar import day_to_date
from src.forecast import store
from src.forecast.pipeline import START, load_config, run_backtest, run_review
from src.data.metrics import milestone_complete_at
from tests.test_historical_invariance import mutate_not_yet_available

CUT_DAY = 168  # long enough at 30% volume for the first calibrated intervals to appear
TABLES = ("forecast_issued", "forecast_withheld", "weekly_summary")


def small_cfg(last_day=CUT_DAY):
    cfg = load_config()
    cfg["quality"].update(min_installs_per_cohort=8, window_min_installs=40)  # 30% volume: cohorts are ~3x smaller
    cfg["backtest"]["last_as_of_day"] = last_day
    return cfg


def read(path):
    c = sqlite3.connect(path)
    try:
        return {t: pd.read_sql_query(f"SELECT * FROM {t} ORDER BY 1, 2", c) for t in TABLES}
    finally:
        c.close()


@pytest.fixture(scope="module")
def backtest(small_dataset, tmp_path_factory):
    out = tmp_path_factory.mktemp("bt")
    run_backtest(small_dataset["db"], out / "f.db", small_cfg())
    return {"store": out / "f.db", "tables": read(out / "f.db"), "dir": out}


def test_run_schedule_is_recorded_separately_from_the_evidence_hash(backtest):
    cfg = small_cfg()
    c = sqlite3.connect(backtest["store"])
    meta = store.read_meta(c)
    assert meta["forecast_evidence_hash"] == store.forecast_evidence_hash(cfg)
    assert json.loads(meta["schedule_json"]) == cfg["backtest"] and meta["schedule_hash"] == store.sha256(cfg["backtest"])
    # the evidence hash does not move with the schedule ...
    later = small_cfg(last_day=CUT_DAY + 70)
    assert store.forecast_evidence_hash(later) == store.forecast_evidence_hash(cfg) and store.sha256(later["backtest"]) != meta["schedule_hash"]
    # ... and every stored summary and forecast carries the EVIDENCE hash, not the schedule hash
    assert set(pd.read_sql_query("SELECT DISTINCT config_hash FROM weekly_summary", c).config_hash) == {meta["forecast_evidence_hash"]}
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        c.execute("UPDATE store_meta SET value = 'x'")
    c.close()


def test_backtest_produces_forecasts_withheld_records_and_summaries(backtest):
    t = backtest["tables"]
    assert len(t["forecast_issued"]) > 100 and len(t["forecast_withheld"]) > 20 and len(t["weekly_summary"]) > 20
    assert not t["forecast_issued"].duplicated(["channel", "install_date"]).any()
    both = set(zip(t["forecast_issued"].channel, t["forecast_issued"].install_date)) & set(zip(t["forecast_withheld"].channel, t["forecast_withheld"].install_date))
    assert not both  # every cohort is attempted once: issued OR withheld


def test_insufficient_history_is_reported_not_papered_over(backtest):
    t = backtest["tables"]
    cfg = small_cfg()["forecast"]
    held = t["forecast_withheld"]
    early = held[held.reason == "insufficient_training_history"]
    assert len(early) > 0 and (early.attempted_as_of <= day_to_date(START, 91)).all()  # only the first ~13 weeks lack history
    detail = json.loads(early.iloc[0].detail_json)
    assert {"channel_training_cohorts", "pooled_training_cohorts", "needed_channel", "needed_pooled"} <= set(detail)
    issued = t["forecast_issued"]
    ok = ((issued.method == "channel_multiplier") & (issued.train_n >= cfg["min_train_cohorts"])) | \
         ((issued.method == "pooled_multiplier") & (issued.train_n >= cfg["min_pooled_cohorts"])) | (issued.method == "zero_d3_fallback")
    assert ok.all()
    assert issued.issued_as_of.min() > early.attempted_as_of.min()  # nothing issued during the earliest, history-less reviews


def test_training_and_calibration_use_only_outcomes_complete_at_issuance(backtest):
    issued = backtest["tables"]["forecast_issued"]
    assert (issued.train_max_outcome_complete_at <= issued.issued_as_of).all()
    cal = issued[issued.interval_status == "available"]
    assert len(cal) > 0 and (cal.calib_max_outcome_complete_at <= cal.issued_as_of).all()
    assert (cal.calib_n >= small_cfg()["forecast"]["cohort_min_calibration"]).all()
    un = issued[issued.interval_status == "unavailable"]
    assert un.interval_lo.isna().all() and un.interval_hi.isna().all() and un.interval_reason.notna().all()


def test_no_cohort_is_forecast_after_its_own_d30_outcome_is_known(backtest, small_dataset):
    issued = backtest["tables"]["forecast_issued"]
    dec = sqlite3.connect(small_dataset["db"])
    done = milestone_complete_at(dec, "2099-01-01", 30).rename(columns={"complete_at": "complete30"})
    dec.close()
    m = issued.merge(done, on=["channel", "install_date"])
    assert (m.complete30 > m.issued_as_of).all()


def test_stored_features_only_contain_d0_d3_information_and_match_raw_data(backtest, small_dataset):
    issued = backtest["tables"]["forecast_issued"].sample(8, random_state=0)
    dec = sqlite3.connect(small_dataset["db"])
    try:
        for r in issued.itertuples(index=False):
            f = json.loads(r.features_json)
            raw = dec.execute(
                "SELECT COALESCE(SUM(d.iap_gross),0) * 0.7 + COALESCE(SUM(d.ad_revenue),0) FROM player_days d JOIN players p ON p.player_id = d.player_id "
                "WHERE p.channel = ? AND p.install_date = ? AND d.day_n <= 3", (r.channel, r.install_date)).fetchone()[0]
            assert f["d0_3_net_revenue"] == pytest.approx(raw)  # same number computed from the full data: nothing late was missing
            assert store.sha256(f) == r.feature_hash
            assert r.point_forecast == pytest.approx(json.loads(r.training_json)["multiplier"] * f["d0_3_net_revenue"]) or r.method == "zero_d3_fallback"
    finally:
        dec.close()


def test_issued_forecasts_are_never_recomputed_even_if_visible_history_is_edited(backtest, small_dataset, tmp_path):
    before = read(backtest["store"])
    store_copy, db_copy = tmp_path / "f.db", tmp_path / "edited.db"
    shutil.copyfile(backtest["store"], store_copy)
    shutil.copyfile(small_dataset["db"], db_copy)
    c = sqlite3.connect(db_copy)  # an UNSUPPORTED edit of already-visible facts (docs/data_contract.md)
    c.execute("UPDATE player_days SET ad_revenue = ad_revenue * 3, iap_gross = iap_gross * 3")
    c.commit()
    c.close()
    dec, st = sqlite3.connect(db_copy), sqlite3.connect(store_copy)
    try:
        for day in (CUT_DAY + 7, CUT_DAY + 14):  # two later reviews on the edited data
            run_review(dec, st, day_to_date(START, day), small_cfg())
        st.commit()
    finally:
        dec.close()
        st.close()
    after = read(store_copy)
    for t in ("forecast_issued", "forecast_withheld"):
        old = before[t]
        kept = after[t].merge(old[["channel", "install_date"]], on=["channel", "install_date"])
        pd.testing.assert_frame_equal(kept.reset_index(drop=True), old.reset_index(drop=True), check_exact=True)
    assert len(after["forecast_issued"]) > len(before["forecast_issued"])  # new reviews added rows; old rows untouched


def test_rerunning_the_same_review_adds_nothing(backtest, small_dataset, tmp_path):
    store_copy = tmp_path / "f.db"
    shutil.copyfile(backtest["store"], store_copy)
    dec, st = sqlite3.connect(small_dataset["db"]), sqlite3.connect(store_copy)
    try:
        log = run_review(dec, st, day_to_date(START, CUT_DAY), small_cfg())  # the last review again
    finally:
        dec.close()
    st.close()
    assert log["issued"] == 0 and log["withheld"] == {} and log["summaries"] == 0
    after = read(store_copy)
    for t in TABLES:
        pd.testing.assert_frame_equal(after[t], backtest["tables"][t], check_exact=True)


def test_historical_invariance_of_forecasts_to_not_yet_available_records(small_dataset, backtest, tmp_path):
    """Reviews up to CUT_DAY recomputed on a copy whose records reported AFTER the cut were altered/deleted/added."""
    db_copy = tmp_path / "mut.db"
    shutil.copyfile(small_dataset["db"], db_copy)
    mutate_not_yet_available(db_copy, day_to_date(START, CUT_DAY))
    run_backtest(db_copy, tmp_path / "f_mut.db", small_cfg())
    a, b = backtest["tables"], read(tmp_path / "f_mut.db")
    # guard against a vacuous pass: the compared stores must actually contain calibrated forecasts and summaries
    assert len(a["forecast_issued"]) > 100 and (a["forecast_issued"].interval_status == "available").any()
    assert len(a["weekly_summary"]) > 50 and (a["weekly_summary"].interval_status == "available").any()
    for t in TABLES:
        pd.testing.assert_frame_equal(a[t], b[t], check_exact=True, obj=t)


def test_control_changing_visible_history_changes_the_forecasts(small_dataset, backtest, tmp_path):
    db_copy = tmp_path / "vis.db"
    shutil.copyfile(small_dataset["db"], db_copy)
    c = sqlite3.connect(db_copy)
    c.execute(f"UPDATE player_days SET ad_revenue = ad_revenue * 2 WHERE day_n <= 3 AND reported_at <= '{day_to_date(START, 60)}'")
    c.commit()
    c.close()
    run_backtest(db_copy, tmp_path / "f_vis.db", small_cfg())
    assert not backtest["tables"]["forecast_issued"].equals(read(tmp_path / "f_vis.db")["forecast_issued"])


def window_dates(start, end):
    return pd.date_range(start, end).strftime("%Y-%m-%d").tolist()


def test_every_summary_accounts_for_every_expected_date_exactly_once(backtest):
    s = backtest["tables"]["weekly_summary"]
    valid = {"forecast", "withheld", "pending_incomplete", "missing", "unaccounted"}
    for r in s.itertuples(index=False):
        acc = json.loads(r.accounting_json)
        dates = [a["install_date"] for a in acc]
        assert dates == window_dates(r.window_start, r.window_end) and len(dates) == r.n_expected_dates  # contiguous, no date skipped
        assert {a["status"] for a in acc} <= valid
        counts = pd.Series([a["status"] for a in acc]).value_counts().to_dict()
        assert counts.get("forecast", 0) == r.n_included and counts.get("withheld", 0) == r.n_withheld
        blocked = counts.get("pending_incomplete", 0) + counts.get("missing", 0) + counts.get("unaccounted", 0)
        assert (r.fully_decision_ready == 1) == (blocked == 0 and json.loads(r.not_ready_reasons_json) == [])
        if r.fully_decision_ready == 0:
            assert r.interval_status == "unavailable"  # a not-ready summary never presents an interval


def test_cumulative_revenue_floor_holds_for_every_forecast_and_summary(backtest):
    t = backtest["tables"]
    f, s = t["forecast_issued"], t["weekly_summary"]
    assert (f.point_forecast >= f.observed_d3_floor - 1e-9).all()
    iv = f[f.interval_status == "available"]
    assert (iv.interval_lo >= iv.observed_d3_floor - 1e-9).all() and (iv.interval_lo <= iv.interval_hi).all()
    assert f[f.interval_status == "unavailable"].interval_floor_adjusted.eq(0).all()
    sv = s[s.interval_status == "available"]
    assert (sv.interval_lo >= sv.observed_d3_floor - 1e-9).all() and (sv.interval_lo <= sv.interval_hi).all()
    assert (s.point_forecast.dropna() >= s.observed_d3_floor.dropna() - 1e-9).all()
    # the floor equals the sum of the included cohorts' stored D0-D3 revenue (summaries are sums of stored cohort forecasts)
    issued = f.set_index(["channel", "install_date"])
    for r in s[s.n_included > 0].itertuples(index=False):
        keys = json.loads(r.included_json)
        assert r.observed_d3_floor == pytest.approx(sum(json.loads(issued.loc[(r.channel, k), "features_json"])["d0_3_net_revenue"] for k in keys))


AS_OF_DAY = 245  # review as of 2025-09-08
BAD_DATE = "2025-08-31"


def summary_after(small_dataset, tmp_path, sql=None, channel="meta"):
    """One review at 2025-09-08 on a copy of the dataset after optionally running `sql` (a data-delivery problem)."""
    db = tmp_path / "copy.db"
    shutil.copyfile(small_dataset["db"], db)
    if sql:
        c = sqlite3.connect(db)
        c.executescript(sql)
        c.commit()
        c.close()
    dec, st = sqlite3.connect(db), store.create_store(tmp_path / "s.db")
    try:
        run_review(dec, st, day_to_date(START, AS_OF_DAY), small_cfg())
        s = pd.read_sql_query("SELECT * FROM weekly_summary WHERE channel = ?", st, params=(channel,)).iloc[0]
    finally:
        dec.close()
        st.close()
    quality = {c["name"]: c for c in json.loads(s.quality_json)}
    return s, {a["install_date"]: a for a in json.loads(s.accounting_json)}, quality


def test_ordinary_reporting_lag_stays_decision_ready(small_dataset, tmp_path):
    for channel in ("meta", "unity"):  # unity has a long declared lag: many later dates pending is still ordinary
        sub = tmp_path / channel
        sub.mkdir()
        s, acc, quality = summary_after(small_dataset, sub, channel=channel)
        assert s.fully_decision_ready == 1 and quality["reporting_completeness"]["passed"] is True
        assert {a["status"] for a in acc.values()} <= {"forecast", "withheld"}  # every window date is a known outcome
        later = quality["reporting_completeness"]["evidence"]["later_pending_dates"]
        assert 0 < len(later) <= quality["reporting_completeness"]["evidence"]["expected_max_later_pending"]


def test_interior_incomplete_date_is_visible_blocks_readiness_and_is_not_treated_as_complete(small_dataset, tmp_path):
    """Reproducer from independent review: delay D3 completeness for meta installed 2025-08-31, review as of 2025-09-08."""
    sql = f"UPDATE cohort_reporting SET complete_at = '2026-01-01' WHERE channel = 'meta' AND install_date = '{BAD_DATE}' AND through_day = 3;"
    s, acc, quality = summary_after(small_dataset, tmp_path, sql)
    assert acc[BAD_DATE]["status"] == "pending_incomplete" and "not final" in acc[BAD_DATE]["note"]
    assert BAD_DATE not in json.loads(s.included_json) and BAD_DATE not in [h["install_date"] for h in json.loads(s.withheld_json)]
    assert s.fully_decision_ready == 0 and any(BAD_DATE in r for r in json.loads(s.not_ready_reasons_json))
    rc = quality["reporting_completeness"]
    assert rc["passed"] is False and rc["evidence"]["interior_incomplete_dates"] == [BAD_DATE]
    assert quality["missing_days"]["passed"] is True  # the cohort EXISTS (partial data); it is not a missing date
    assert s.interval_status == "unavailable" and s.interval_reason.startswith("window_not_fully_decision_ready")
    # the other checks do not silently treat the partial cohort as complete
    assert BAD_DATE in quality["cpi_anomaly"]["evidence"]["excluded_incomplete_dates"]
    conc = quality["revenue_concentration"]["evidence"]
    assert BAD_DATE in conc["excluded_dates"] and BAD_DATE not in conc["eligible_dates"]
    sub = tmp_path / "clean"
    sub.mkdir()
    clean, _, _ = summary_after(small_dataset, sub)
    assert clean.fully_decision_ready == 1  # same review without the delayed metadata is fully decision-ready


def test_payer_concentration_sql_excludes_players_of_incomplete_cohorts(small_dataset, tmp_path):
    db = tmp_path / "c.db"
    shutil.copyfile(small_dataset["db"], db)
    c = sqlite3.connect(db)
    c.execute(f"UPDATE cohort_reporting SET complete_at = '2026-01-01' WHERE channel = 'meta' AND install_date = '{BAD_DATE}' AND through_day = 3")
    c.commit()
    ids = {r[0] for r in c.execute(f"SELECT player_id FROM players WHERE channel = 'meta' AND install_date = '{BAD_DATE}'")}
    from src.data.metrics import payer_gross
    got = set(payer_gross(c, day_to_date(START, AS_OF_DAY), "meta", "2025-08-28", "2025-09-03").player_id)
    c.close()
    assert got and not (got & ids)


def test_genuinely_missing_date_is_missing_not_zero_and_blocks_readiness(small_dataset, tmp_path):
    sql = f"""
    DELETE FROM player_days WHERE player_id IN (SELECT player_id FROM players WHERE channel = 'meta' AND install_date = '{BAD_DATE}');
    DELETE FROM players WHERE channel = 'meta' AND install_date = '{BAD_DATE}';
    DELETE FROM spend_daily WHERE channel = 'meta' AND spend_date = '{BAD_DATE}';
    DELETE FROM cohort_reporting WHERE channel = 'meta' AND install_date = '{BAD_DATE}';
    """
    s, acc, quality = summary_after(small_dataset, tmp_path, sql)
    assert acc[BAD_DATE]["status"] == "missing"
    assert quality["missing_days"]["passed"] is False and quality["missing_days"]["evidence"]["missing_dates"] == [BAD_DATE]
    assert quality["reporting_completeness"]["evidence"]["interior_incomplete_dates"] == []  # missing is not "pending"
    assert s.fully_decision_ready == 0 and s.interval_status == "unavailable"


def test_zero_activity_cohort_is_not_missing_and_does_not_block_readiness(small_dataset, tmp_path):
    """Present, complete cohort whose players have no activity rows: a KNOWN zero, accounted as withheld/forecast."""
    sql = f"DELETE FROM player_days WHERE player_id IN (SELECT player_id FROM players WHERE channel = 'meta' AND install_date = '{BAD_DATE}');"
    s, acc, quality = summary_after(small_dataset, tmp_path, sql)
    assert acc[BAD_DATE]["status"] in ("withheld", "forecast")
    assert quality["missing_days"]["passed"] is True and s.fully_decision_ready == 1


def test_summaries_document_their_cohorts_and_stay_separate_from_cohort_forecasts(backtest):
    t = backtest["tables"]
    issued = t["forecast_issued"].set_index(["channel", "install_date"])
    for s in t["weekly_summary"].itertuples(index=False):
        keys = json.loads(s.included_json)
        assert len(keys) == s.n_included and all(s.window_start <= k <= s.window_end for k in keys)
        if keys:
            expected = sum(issued.loc[(s.channel, k), "point_forecast"] for k in keys)
            assert s.point_forecast == pytest.approx(expected)  # a pure sum of stored cohort forecasts, no new model
        assert all(r["install_date"] >= s.window_start for r in json.loads(s.withheld_json))
        assert {c["name"] for c in json.loads(s.quality_json)} >= {"window_min_installs", "missing_days", "reporting_completeness",
                                                                  "revenue_concentration", "cpi_anomaly", "tracking_gap"}
