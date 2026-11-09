"""Issue cohort forecasts and weekly channel summaries at one review date, and run the chronological backtest.

Rules enforced here:
  * Every input is read through the availability layer at `as_of` (via src.data.metrics), nothing else.
  * Each cohort is attempted ONCE, at the first review where it is decision-ready. It is either issued (stored, never
    recomputed) or withheld with an explicit reason (also stored). A cohort whose D30 outcome is already complete is not forecast.
  * Training uses only cohorts whose D30 outcome was complete and reported at `as_of`, and that pass the quality checks.
  * Intervals use only earlier issued forecasts whose outcomes were complete at `as_of`.
Weekly summaries are separate from cohort forecasts: the summary for a channel covers the 7 consecutive install dates ending
at the channel's latest decision-ready install date; the included cohorts are stored with the summary.
"""
import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.calendar import day_to_date
from src.data.metrics import cohort_metrics, cohort_outcomes, daily_activity, milestone_complete_at, payer_gross
from src.forecast import baseline, intervals, store
from src.quality import checks

ROOT = Path(__file__).resolve().parents[2]
START = "2025-01-06"
FEATURES = ["installs_reported", "spend", "cpi", "d1_retention", "d3_retention", "d0_3_net_revenue", "d0_3_gross_iap",
            "d0_3_ad_revenue", "payer_rate_d0_3", "arpdau_d0_3", "sessions_per_active_user", "levels_per_active_user",
            "d3_net_per_install", "base_d3_net_per_install", "base_d1"]


def load_config(path=None) -> dict:
    with open(path or ROOT / "config" / "forecast.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _clean(v):
    if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def _dates(end: str, n: int):
    e = date.fromisoformat(end)
    return [(e - timedelta(days=i)).isoformat() for i in range(n - 1, -1, -1)]


# ---------------------------------------------------------------- one review


def run_review(dec: sqlite3.Connection, st: sqlite3.Connection, as_of: str, cfg: dict) -> dict:
    """Issue forecasts for newly decision-ready cohorts and write one summary per paid channel. Returns a small log."""
    chash = store.forecast_evidence_hash(cfg)
    q, f = cfg["quality"], cfg["forecast"]

    m = cohort_metrics(dec, as_of)
    paid = m[m.channel != "organic"]
    ann = checks.annotate_cohorts(paid, q)
    out30 = cohort_outcomes(dec, as_of, 30)[["channel", "install_date", "net_revenue"]].rename(columns={"net_revenue": "y30"})
    comp30 = milestone_complete_at(dec, as_of, 30).rename(columns={"complete_at": "complete30"})
    comp3 = milestone_complete_at(dec, as_of, 3)
    ready = ann[ann.d3_complete == 1]

    # training pool: eligible cohorts whose D30 outcome is complete NOW
    mature = ready.merge(out30, on=["channel", "install_date"]).merge(comp30, on=["channel", "install_date"])
    pool = mature[mature.exclusion_reason.isna() & ~mature.zero_d3].copy()
    pool["y3"], pool["installs"] = pool.d0_3_net_revenue, pool.installs_reported
    pools = {ch: g[["install_date", "y3", "y30", "installs", "complete30"]] for ch, g in pool.groupby("channel")}
    channels = sorted(paid.channel.unique())
    training = {ch: baseline.choose_training(ch, pools, cfg) for ch in channels}

    # interval calibration inputs (earlier issued forecasts only)
    earlier = pd.read_sql_query(
        "SELECT channel, install_date, point_forecast AS point FROM forecast_issued WHERE method = 'channel_multiplier' AND issued_as_of < ?",
        st, params=(as_of,))
    calib = {}
    for ch in channels:
        calib[ch] = intervals.matured_ratios(earlier[earlier.channel == ch], out30, comp30, f["cohort_calibration_window"])

    attempted = store.attempted_keys(st)
    matured_keys = set(zip(out30.channel, out30.install_date))
    log = {"as_of": as_of, "issued": 0, "withheld": {}, "summaries": 0}
    for r in ready.itertuples(index=False):
        key = (r.channel, r.install_date)
        if key in attempted:
            continue
        feats = {k: _clean(getattr(r, k)) for k in FEATURES}
        feats["zero_d3"] = bool(r.zero_d3)
        reason, detail = None, {}
        if key in matured_keys:
            reason, detail = "outcome_already_available", {"note": "D30 outcome already complete at this review, so it is not a forecast"}
        elif pd.notna(r.exclusion_reason):  # NaN is truthy, so test explicitly
            reason, detail = r.exclusion_reason, r.exclusion_detail
        else:
            level, info, why = training[r.channel]
            if info is None:
                reason, detail = "insufficient_training_history", why
            else:
                fc = baseline.forecast_cohort(r.d0_3_net_revenue, int(r.installs_reported), level, info)
                if fc["point"] is None:
                    reason, detail = "no_forecast_value", {"note": "training set has no installs to derive a value per install"}
        if reason:
            st.execute("INSERT INTO forecast_withheld VALUES (?,?,?,?,?)", (r.channel, r.install_date, as_of, reason, store.canonical_json(detail)))
            log["withheld"][reason] = log["withheld"].get(reason, 0) + 1
            continue
        # Cumulative D30 net revenue cannot be below the D0-D3 net revenue already observed (revenue is non-negative).
        floor = float(feats["d0_3_net_revenue"] or 0.0)
        point = max(fc["point"], floor)
        if fc["method"] == "channel_multiplier":
            ratios, calib_last = calib[r.channel]
            iv = intervals.interval_from_ratios(ratios, point, f["interval_alpha"], f["cohort_min_calibration"])
        else:
            iv = {"status": "unavailable", "lo": None, "hi": None, "calib_n": 0, "reason": f"non_standard_method: {fc['method']} has no calibrated interval"}
            calib_last = None
        lo, hi, iv_adjusted = intervals.apply_floor(iv["lo"], iv["hi"], floor)
        row = {
            "channel": r.channel, "install_date": r.install_date, "issued_as_of": as_of, "method": fc["method"],
            "features_json": store.canonical_json(feats), "feature_hash": store.sha256(feats),
            "training_json": store.canonical_json(dict(info, level=level)), "train_n": info["n_cohorts"],
            "train_max_outcome_complete_at": info["max_outcome_complete_at"],
            "point_forecast": point, "reference_forecast": fc["reference"],
            "observed_d3_floor": floor, "point_floor_adjusted": int(point != fc["point"]),
            "interval_status": iv["status"], "interval_lo": lo, "interval_hi": hi, "interval_floor_adjusted": int(iv_adjusted),
            "interval_reason": iv["reason"], "calib_n": iv["calib_n"],
            "calib_max_outcome_complete_at": calib_last if iv["status"] == "available" else None, "config_hash": chash,
        }
        st.execute(f"INSERT INTO forecast_issued ({','.join(row)}) VALUES ({','.join('?' * len(row))})", tuple(row.values()))
        log["issued"] += 1
    st.commit()

    for ch in channels:
        if _weekly_summary(dec, st, as_of, cfg, chash, ch, m, ann, ready, out30, comp30, comp3):
            log["summaries"] += 1
    st.commit()
    return log


# ---------------------------------------------------------------- weekly channel summary


def _weekly_summary(dec, st, as_of, cfg, chash, ch, m, ann, ready, out30, comp30, comp3) -> bool:
    q, f = cfg["quality"], cfg["forecast"]
    if st.execute("SELECT 1 FROM weekly_summary WHERE channel = ? AND as_of = ?", (ch, as_of)).fetchone():
        return False  # already written for this review: summaries are immutable and issued once
    rdy = ready[ready.channel == ch]
    if rdy.empty:
        return False
    last_ready = rdy.install_date.max()
    chm = m[m.channel == ch].set_index("install_date")
    # expected dates: the 7 consecutive install dates ending at the latest decision-ready one, but not before the channel's
    # first visible data (dates before the history starts are not "missing")
    window = [d for d in _dates(last_ready, f["summary_days"]) if d >= chm.index.min()]
    w0, w1 = window[0], window[-1]

    issued = pd.read_sql_query(
        "SELECT install_date, method, point_forecast, features_json FROM forecast_issued WHERE channel = ? AND install_date BETWEEN ? AND ?",
        st, params=(ch, w0, w1))
    held = pd.read_sql_query(
        "SELECT install_date, reason FROM forecast_withheld WHERE channel = ? AND install_date BETWEEN ? AND ?", st, params=(ch, w0, w1))
    feats = [json.loads(x) for x in issued.features_json]
    installs_inc = int(sum(x["installs_reported"] for x in feats))
    spend_inc = float(sum(x["spend"] for x in feats))
    point = float(issued.point_forecast.sum()) if len(issued) else None
    d3_floor = float(sum((x["d0_3_net_revenue"] or 0.0) for x in feats)) if feats else None
    included = sorted(issued.install_date.tolist())

    # ---- account for EVERY expected install date: forecast | withheld | pending_incomplete | missing | unaccounted
    issued_set, held_map = set(issued.install_date), dict(zip(held.install_date, held.reason))
    accounting, interior_incomplete, missing_dates, unaccounted = [], [], [], []
    for d in window:
        if d in issued_set:
            accounting.append({"install_date": d, "status": "forecast"})
        elif d in held_map:
            accounting.append({"install_date": d, "status": "withheld", "reason": held_map[d]})
        elif d in chm.index and chm.loc[d, "d3_complete"] == 1:
            accounting.append({"install_date": d, "status": "unaccounted", "note": "complete cohort with no stored forecast or withheld record"})
            unaccounted.append(d)
        elif d in chm.index:  # cohort known (players or spend visible) but D3 not declared complete: partial data, NOT zero
            accounting.append({"install_date": d, "status": "pending_incomplete", "installs_reported_so_far": int(chm.loc[d, "installs_reported"]),
                               "note": "D3 not declared complete inside the window: install count is not final (unknown, not zero)"})
            interior_incomplete.append(d)
        else:  # no players and no spend visible at all: data missing (distinct from a complete cohort with zero activity)
            accounting.append({"install_date": d, "status": "missing", "note": "no cohort data visible for this date"})
            missing_dates.append(d)
    complete_dates = [d for d in window if d in chm.index and chm.loc[d, "d3_complete"] == 1]

    # ---- window-level quality checks
    present = set(chm.index)
    later_dates = []  # install dates after the latest decision-ready one, before as_of: reporting delay, not zero
    d = date.fromisoformat(last_ready) + timedelta(days=1)
    while d.isoformat() < as_of:
        later_dates.append(d.isoformat())
        d += timedelta(days=1)
    lag = int(max((pd.Timestamp(c) - pd.Timestamp(i)).days - 3 for c, i in zip(comp3[comp3.channel == ch].complete_at, comp3[comp3.channel == ch].install_date)))
    gross = payer_gross(dec, as_of, ch, w0, w1).iap_gross.tolist()  # SQL only includes D3-complete cohorts
    completeness = checks.check_reporting_completeness(interior_incomplete, later_dates, w1, lag)
    qual = [
        checks.check_window_min_installs(installs_inc, q["window_min_installs"]),
        checks.check_missing_days(window, present),
        completeness,
        checks.check_revenue_concentration(gross, q["concentration_top_share"], complete_dates, window),
        checks.check_window_cpi(ann[ann.channel == ch], window, q),
        checks.check_tracking_gap(daily_activity(dec, as_of, ch), lag, as_of, q),
    ]
    not_ready = []
    if missing_dates:
        not_ready.append(f"{len(missing_dates)} expected install date(s) missing: {missing_dates}")
    if interior_incomplete:
        not_ready.append(f"{len(interior_incomplete)} install date(s) inside the window not decision-ready: {interior_incomplete}")
    if unaccounted:
        not_ready.append(f"{len(unaccounted)} complete date(s) without a forecast or withheld record: {unaccounted}")
    if not completeness.passed and not interior_incomplete:
        not_ready.append(completeness.reason)
    fully_ready = not not_ready

    # ---- interval from EARLIER fully decision-ready summaries that have matured
    if point is None or not included:
        iv = {"status": "unavailable", "lo": None, "hi": None, "calib_n": 0, "reason": "no_included_cohorts"}
    elif not fully_ready:
        iv = {"status": "unavailable", "lo": None, "hi": None, "calib_n": 0, "reason": "window_not_fully_decision_ready: " + "; ".join(not_ready)}
    else:
        prev = pd.read_sql_query("SELECT as_of, included_json, point_forecast FROM weekly_summary "
                                 "WHERE channel = ? AND as_of < ? AND point_forecast > 0 AND fully_decision_ready = 1",
                                 st, params=(ch, as_of))
        actual = out30[out30.channel == ch].set_index("install_date").y30
        ratios = []
        for p in prev.sort_values("as_of").itertuples(index=False):
            keys = json.loads(p.included_json)
            if keys and all(k in actual.index for k in keys):
                ratios.append(float(actual.loc[keys].sum() / p.point_forecast))
        iv = intervals.interval_from_ratios(ratios[-f["summary_calibration_window"]:], point, f["interval_alpha"], f["summary_min_calibration"])

    lo, hi, iv_adjusted = intervals.apply_floor(iv["lo"], iv["hi"], d3_floor or 0.0)  # floor = sum of included cohorts' observed D0-D3 revenue
    row = {
        "channel": ch, "as_of": as_of, "window_start": w0, "window_end": w1,
        "included_json": json.dumps(included), "withheld_json": json.dumps(sorted(held.to_dict("records"), key=lambda r: r["install_date"])),
        "accounting_json": store.canonical_json(accounting), "fully_decision_ready": int(fully_ready),
        "not_ready_reasons_json": json.dumps(not_ready), "n_expected_dates": len(window),
        "n_included": len(issued), "n_withheld": len(held), "installs_included": installs_inc, "spend_included": spend_inc,
        "point_forecast": point, "observed_d3_floor": d3_floor,
        "interval_status": iv["status"], "interval_lo": lo, "interval_hi": hi, "interval_floor_adjusted": int(iv_adjusted),
        "interval_reason": iv["reason"], "calib_n": iv["calib_n"],
        "quality_json": store.canonical_json([c.to_dict() for c in qual]), "config_hash": chash,
    }
    st.execute(f"INSERT INTO weekly_summary ({','.join(row)}) VALUES ({','.join('?' * len(row))})", tuple(row.values()))
    return True


# ---------------------------------------------------------------- backtest


def run_backtest(dec_path, store_path, cfg=None, progress=None) -> list:
    """Chronological backtest: one review per Monday from first_as_of_day to last_as_of_day. Returns the per-review log."""
    cfg = cfg or load_config()
    b = cfg["backtest"]
    dec = sqlite3.connect(dec_path)
    st = store.create_store(store_path, schedule=b, evidence_hash=store.forecast_evidence_hash(cfg))  # schedule recorded separately from the evidence hash
    logs = []
    try:
        for day in range(b["first_as_of_day"], b["last_as_of_day"] + 1, b["review_every_days"]):
            log = run_review(dec, st, day_to_date(START, day), cfg)
            log["as_of_day"] = day
            logs.append(log)
            if progress:
                progress(log)
    finally:
        dec.close()
        st.close()
    return logs
