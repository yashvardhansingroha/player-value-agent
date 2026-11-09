"""Score issued forecasts against matured D30 outcomes. Scoring is post-hoc evaluation: it reads actual outcomes through
the same availability layer at a scoring date, and only outcomes that are complete and reported at that date count.

Formulas (all over the scored cohorts / summaries):
  WAPE     = sum(|actual - forecast|) / sum(actual)           (total-revenue WAPE; install weights cancel, see PLAN)
  bias     = sum(forecast - actual) / sum(actual)             (positive = over-forecast)
  coverage = share of scored items whose actual lies inside [interval_lo, interval_hi]; only items that HAVE an interval
  interval_width_over_forecast_mean / _median = mean / median of (hi - lo) / point forecast over the items that have an interval. The un-suffixed
      `interval_width_over_forecast` is a LEGACY alias of the MEDIAN (the Phase 2 report reads it as "median width / forecast"); new code uses the suffixed keys.
"""
import json
import sqlite3

import numpy as np
import pandas as pd

from src.data.metrics import cohort_outcomes


def metrics(df: pd.DataFrame, forecast_col="point_forecast", with_interval=True) -> dict:
    """df needs actual, <forecast_col>, interval_status, interval_lo, interval_hi."""
    if df.empty:
        return {"n": 0}
    a, p = df.actual.to_numpy(), df[forecast_col].to_numpy()
    out = {
        "n": int(len(df)),
        "sum_actual": float(a.sum()),
        "sum_forecast": float(p.sum()),
        # undefined (None), not infinite, when the scored items have zero actual revenue in total
        "wape": float(np.abs(a - p).sum() / a.sum()) if a.sum() > 0 else None,
        "bias": float((p - a).sum() / a.sum()) if a.sum() > 0 else None,
    }
    if not with_interval:
        return out
    iv = df[df.interval_status == "available"]
    out["n_with_interval"] = int(len(iv))
    if len(iv):
        out["coverage"] = float(((iv.actual >= iv.interval_lo) & (iv.actual <= iv.interval_hi)).mean())
        width = (iv.interval_hi - iv.interval_lo) / iv.point_forecast
        out["interval_width_over_forecast_mean"] = float(width.mean())
        out["interval_width_over_forecast_median"] = float(width.median())
        # LEGACY key, kept because the Phase 2 report reads it (and labels it "median width / forecast"). It is the MEDIAN, not the mean: use the two explicit keys above.
        out["interval_width_over_forecast"] = out["interval_width_over_forecast_median"]
    return out


def _ranges(df: pd.DataFrame, cols) -> dict:
    return {c: [str(df[c].min()), str(df[c].max())] for c in cols} if len(df) else {}


def score_cohorts(st: sqlite3.Connection, dec: sqlite3.Connection, scoring_as_of: str, review_dates=None) -> dict:
    issued = pd.read_sql_query("SELECT * FROM forecast_issued", st)
    held = pd.read_sql_query("SELECT * FROM forecast_withheld", st)
    out30 = cohort_outcomes(dec, scoring_as_of, 30)[["channel", "install_date", "net_revenue"]].rename(columns={"net_revenue": "actual"})
    scored = issued.merge(out30, on=["channel", "install_date"], how="left")
    pending = scored[scored.actual.isna()]
    scored = scored[scored.actual.notna()].copy()
    # periods: four groups of equal numbers of review dates (generic, not tied to any planted event)
    dates = sorted(scored.issued_as_of.unique())
    cuts = np.array_split(np.array(dates), 4) if dates else []
    period = {d: f"P{i + 1} ({c[0]}..{c[-1]})" for i, c in enumerate(cuts) for d in c}
    scored["period"] = scored.issued_as_of.map(period)

    result = {
        "scoring_as_of": scoring_as_of,
        "issued": int(len(issued)),
        "scored": int(len(scored)),
        "pending_outcome_not_in_storage": int(len(pending)),
        "ranges": _ranges(scored, ["install_date", "issued_as_of"]),
        "overall": metrics(scored),
        "overall_with_interval_only": metrics(scored[scored.interval_status == "available"]),
        "reference_trailing_value_per_install_ignoring_d3": metrics(scored, forecast_col="reference_forecast", with_interval=False),
        "by_method": {k: metrics(g) for k, g in scored.groupby("method")},
        "by_channel": {k: metrics(g) for k, g in scored.groupby("channel")},
        "by_period": {k: metrics(g) for k, g in scored.groupby("period")},
        "by_channel_with_interval": {k: metrics(g[g.interval_status == "available"]) for k, g in scored.groupby("channel")},
        "by_period_with_interval": {k: metrics(g[g.interval_status == "available"]) for k, g in scored.groupby("period")},
    }
    result["interval_unavailable_reasons"] = scored[scored.interval_status == "unavailable"].interval_reason.str.split(":").str[0].value_counts().to_dict()
    # structural floor (cumulative D30 >= observed D3): how often it changed anything. It does not widen intervals.
    result["floor_adjustments"] = {
        "point_forecasts_raised": int(scored.point_floor_adjusted.sum()),
        "intervals_adjusted": int(scored.interval_floor_adjusted.sum()),
        "actual_below_observed_d3_floor": int((scored.actual < scored.observed_d3_floor - 1e-9).sum()),
        "any_interval_unordered": int((scored.interval_lo > scored.interval_hi).sum()),
    }

    # withheld cohorts: reasons, and the actual D30 revenue they carried (context for how much was excluded)
    hw = held.merge(out30, on=["channel", "install_date"], how="left")
    by_reason = {}
    for reason, g in hw.groupby("reason"):
        by_reason[reason] = {"cohorts": int(len(g)), "matured_actual_d30_net_revenue": float(g.actual.sum()),
                             "matured_cohorts": int(g.actual.notna().sum()),
                             "by_channel": g.channel.value_counts().to_dict()}
    result["withheld"] = by_reason
    result["withheld_total"] = int(len(held))
    result["share_of_matured_d30_revenue_withheld"] = float(hw.actual.sum() / (hw.actual.sum() + scored.actual.sum()))
    if review_dates is not None:
        result["reviews"] = len(review_dates)
    return result


def score_subset(st: sqlite3.Connection, dec: sqlite3.Connection, scoring_as_of: str, keys) -> dict:
    """Metrics for issued forecasts restricted to (channel, install_date) keys; used by the labelled sensitivity run."""
    issued = pd.read_sql_query("SELECT * FROM forecast_issued", st)
    out30 = cohort_outcomes(dec, scoring_as_of, 30)[["channel", "install_date", "net_revenue"]].rename(columns={"net_revenue": "actual"})
    m = issued.merge(out30, on=["channel", "install_date"], how="inner")
    m = m[[(c, d) in keys for c, d in zip(m.channel, m.install_date)]]
    return metrics(m)


def score_summaries(st: sqlite3.Connection, dec: sqlite3.Connection, scoring_as_of: str) -> dict:
    summ = pd.read_sql_query("SELECT * FROM weekly_summary", st)
    out30 = cohort_outcomes(dec, scoring_as_of, 30)[["channel", "install_date", "net_revenue"]]
    actual = out30.set_index(["channel", "install_date"]).net_revenue
    # a summary stores no method of its own: it adds up its included cohorts' forecasts, so its method is the cohorts' method, or 'mixed' if they differ
    methods = pd.read_sql_query("SELECT channel, install_date, method FROM forecast_issued", st).set_index(["channel", "install_date"]).method
    rows, pending, empty = [], 0, 0
    for s in summ.itertuples(index=False):
        keys = json.loads(s.included_json)
        if not keys or s.point_forecast is None or pd.isna(s.point_forecast):
            empty += 1
            continue
        idx = [(s.channel, k) for k in keys]
        if not all(i in actual.index for i in idx):
            pending += 1
            continue
        d = s._asdict()
        ms = {methods.get(i) for i in idx} - {None}
        d["method"] = ms.pop() if len(ms) == 1 else ("mixed" if ms else "unknown")
        d["actual"] = float(actual.loc[idx].sum())
        d["flagged"] = any(not c["passed"] and c["evaluable"] and c["name"] != "reporting_completeness"
                           for c in json.loads(s.quality_json))
        rows.append(d)
    df = pd.DataFrame(rows)
    if df.empty:
        return {"scored": 0, "pending": pending, "empty": empty}
    dates = sorted(df.as_of.unique())  # the same convention as the cohort scorer: four groups of equal numbers of review dates
    period = {d: f"P{i + 1} ({c[0]}..{c[-1]})" for i, c in enumerate(np.array_split(np.array(dates), 4)) for d in c}
    df["period"] = df.as_of.map(period)
    res = {
        "summaries": int(len(summ)), "scored": int(len(df)), "pending_not_mature": pending, "empty_no_included_cohorts": empty,
        "ranges": _ranges(df, ["as_of", "window_start", "window_end"]),
        "overall": metrics(df),
        "by_channel": {k: metrics(g) for k, g in df.groupby("channel")},
        "by_period": {k: metrics(g) for k, g in df.groupby("period")},
        "by_method": {k: metrics(g) for k, g in df.groupby("method")},
        "flagged_by_quality_check": metrics(df[df.flagged]),
        "unflagged": metrics(df[~df.flagged]),
        "fully_decision_ready": metrics(df[df.fully_decision_ready == 1]),
        "not_fully_decision_ready": metrics(df[df.fully_decision_ready == 0]),
        "summaries_not_fully_decision_ready_total": int((summ.fully_decision_ready == 0).sum()),
        "floor_adjustments": {"intervals_adjusted": int(df.interval_floor_adjusted.sum()),
                              "actual_below_observed_d3_floor": int((df.actual < df.observed_d3_floor - 1e-9).sum()),
                              "any_interval_unordered": int((df.interval_lo > df.interval_hi).sum())},
    }
    return res
