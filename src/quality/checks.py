"""Data-quality and reporting checks. Each check returns a CheckResult {name, passed, reason, evidence}.

`passed = True` means "no problem found". `evaluable = False` means the check could not be evaluated (for example too few
prior cohorts to form a baseline); it is reported as such and never silently treated as a pass of real evidence.
Nothing here reads hidden truth or scenario files. All inputs come from availability-filtered decision data.
"""
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class CheckResult:
    name: str
    passed: bool
    reason: str
    evidence: dict = field(default_factory=dict)
    evaluable: bool = True

    def to_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "evaluable": self.evaluable, "reason": self.reason, "evidence": self.evidence}


def _num(x):
    """JSON-safe number or None."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return float(x)


# ------------------------------------------------------------------ cohort-level checks


def check_decision_ready(d3_complete) -> CheckResult:
    ok = bool(d3_complete == 1)
    return CheckResult("decision_ready", ok, "D3 milestone complete" if ok else "D3 not yet declared complete: not decision-ready",
                       {"d3_complete": int(d3_complete)})


def check_missing_spend(spend) -> CheckResult:
    ok = spend is not None and not (isinstance(spend, float) and math.isnan(spend))
    return CheckResult("missing_spend", ok, "spend row visible" if ok else "no visible spend row for this cohort date", {"spend": _num(spend)})


def check_min_installs(installs, minimum) -> CheckResult:
    ok = installs >= minimum
    return CheckResult("min_installs", bool(ok), f"{int(installs)} installs" + ("" if ok else f" is below the minimum {minimum}"),
                       {"installs": int(installs), "minimum": minimum})


def check_zero_d3_revenue(d3_net) -> CheckResult:
    ok = d3_net is not None and not (isinstance(d3_net, float) and math.isnan(d3_net)) and d3_net > 0
    return CheckResult("zero_d3_revenue", bool(ok), "D0-D3 net revenue is positive" if ok else "D0-D3 net revenue is zero",
                       {"d0_3_net_revenue": _num(d3_net)})


def check_revenue_spike(per_install, baseline, ratio_limit) -> CheckResult:
    if baseline is None or np.isnan(baseline) or baseline <= 0:
        return CheckResult("revenue_spike", True, "not evaluable: too few prior cohorts for a baseline", {"baseline": _num(baseline)}, evaluable=False)
    ratio = per_install / baseline
    ok = ratio <= ratio_limit
    return CheckResult("revenue_spike", bool(ok), f"D0-D3 net revenue per install is {ratio:.2f} x the prior median" + ("" if ok else f" (limit {ratio_limit})"),
                       {"per_install": _num(per_install), "baseline_median": _num(baseline), "ratio": _num(ratio), "limit": ratio_limit})


def check_low_activity(d1_retention, baseline, ratio_limit) -> CheckResult:
    if baseline is None or np.isnan(baseline) or baseline <= 0:
        return CheckResult("low_activity", True, "not evaluable: too few prior cohorts for a baseline", {"baseline": _num(baseline)}, evaluable=False)
    ratio = d1_retention / baseline
    ok = ratio >= ratio_limit
    return CheckResult("low_activity", bool(ok), f"D1 retention is {ratio:.2f} x the prior median" + ("" if ok else f" (floor {ratio_limit}); possible tracking gap"),
                       {"d1_retention": _num(d1_retention), "baseline_median": _num(baseline), "ratio": _num(ratio), "floor": ratio_limit})


# exclusion reasons in priority order: the first failing one is reported
EXCLUSION_PRIORITY = ["missing_spend", "no_installs", "below_min_installs", "low_activity", "revenue_spike"]


def annotate_cohorts(cohorts: pd.DataFrame, q: dict) -> pd.DataFrame:
    """Add per-cohort quality columns to cohort_metrics output (paid channels). Baselines are rolling medians of PRIOR
    cohorts only (shifted by one), so a cohort never forms its own baseline.
    Adds: d3_net_per_install, base_d3_net_per_install, base_d1, zero_d3, exclusion_reason, exclusion_detail."""
    df = cohorts.sort_values(["channel", "install_date"]).reset_index(drop=True).copy()
    df["d3_net_per_install"] = np.where(df.installs_reported > 0, df.d0_3_net_revenue / df.installs_reported.replace(0, np.nan), np.nan)
    w, mp = q["baseline_window_cohorts"], q["baseline_min_cohorts"]
    grp = df.groupby("channel", sort=False)
    df["base_d3_net_per_install"] = grp["d3_net_per_install"].transform(lambda s: s.shift(1).rolling(w, min_periods=mp).median())
    df["base_d1"] = grp["d1_retention"].transform(lambda s: s.shift(1).rolling(w, min_periods=mp).median())
    reasons, details, zero = [], [], []
    for r in df.itertuples(index=False):
        reason, detail = None, None
        zero.append(bool(r.d3_complete == 1 and r.installs_reported > 0 and (r.d0_3_net_revenue == 0)))
        if r.d3_complete == 1:
            checks = {
                "missing_spend": check_missing_spend(None if pd.isna(r.spend) else r.spend),
                "no_installs": CheckResult("no_installs", r.installs_reported > 0, "installs reported" if r.installs_reported > 0 else "no installs reported", {"installs": int(r.installs_reported)}),
                "below_min_installs": check_min_installs(r.installs_reported, q["min_installs_per_cohort"]),
            }
            if r.installs_reported > 0:
                checks["low_activity"] = check_low_activity(r.d1_retention, r.base_d1, q["low_activity_ratio"])
                checks["revenue_spike"] = check_revenue_spike(r.d3_net_per_install, r.base_d3_net_per_install, q["spike_ratio"])
            for name in EXCLUSION_PRIORITY:
                c = checks.get(name)
                if c is not None and not c.passed:
                    reason, detail = name, c.to_dict()
                    break
        reasons.append(reason)
        details.append(detail)
    df["zero_d3"] = zero
    df["exclusion_reason"] = reasons
    df["exclusion_detail"] = details
    return df


# ------------------------------------------------------------------ channel / trailing-window checks


def check_missing_days(expected_dates, present_dates) -> CheckResult:
    missing = sorted(set(expected_dates) - set(present_dates))
    return CheckResult("missing_days", not missing, "no missing days" if not missing else f"{len(missing)} expected day(s) have no cohort data",
                       {"missing_dates": missing})


def check_reporting_completeness(interior_incomplete_dates, later_pending_dates, window_end, declared_lag_days) -> CheckResult:
    """Two separate questions.
    1. INSIDE the window: every install date up to the window end should be decision-ready (D3 complete). A date whose cohort
       exists but is not complete is UNEXPECTED here (its data is partial, not zero) and fails the check.
    2. AFTER the window end: the most recent install dates are always still pending (a cohort is decision-ready 3 + declared lag
       days after it installs, so about 2 + lag dates sit between the window end and the review date). That is acceptable up to
       2 + lag dates; more means reporting is later than declared."""
    interior, later = sorted(interior_incomplete_dates), sorted(later_pending_dates)
    expected_max = 2 + declared_lag_days
    ok = not interior and len(later) <= expected_max
    problems = []
    if interior:
        problems.append(f"{len(interior)} install date(s) INSIDE the window are not decision-ready (partial data, not zero)")
    if len(later) > expected_max:
        problems.append(f"{len(later)} later install dates pending, more than the {expected_max} the declared lag explains")
    reason = ("; ".join(problems) if problems else
              f"every window date is decision-ready; {len(later)} later install date(s) pending (up to {expected_max} expected from the declared lag), excluded, not counted as zero")
    return CheckResult("reporting_completeness", ok, reason,
                       {"interior_incomplete_dates": interior, "later_pending_dates": later, "window_end": window_end,
                        "expected_max_later_pending": expected_max})


def check_window_min_installs(installs, minimum) -> CheckResult:
    ok = installs >= minimum
    return CheckResult("window_min_installs", bool(ok), f"{int(installs)} installs in window" + ("" if ok else f" is below {minimum}: insufficient cohort size"),
                       {"installs": int(installs), "minimum": minimum})


def check_revenue_concentration(gross_by_payer, top_share_limit, eligible_dates=None, window_dates=None) -> CheckResult:
    """Share of D0-D3 gross IAP from the top 1% of payers (at least one payer).
    ELIGIBLE POPULATION: only cohorts whose D3 milestone is complete (the caller's SQL joins the completeness metadata).
    `eligible_dates` are the window install dates that qualify and `window_dates` all expected dates; the evidence records both so
    a partial population is visible. With no eligible cohort the check is unavailable, never silently 'passed'."""
    eligible = None if eligible_dates is None else sorted(eligible_dates)
    pop = {} if eligible is None else {"eligible_dates": eligible, "window_dates_expected": len(window_dates or []),
                                      "excluded_dates": sorted(set(window_dates or []) - set(eligible))}
    if eligible is not None and len(eligible) == 0:
        return CheckResult("revenue_concentration", True, "not evaluable: no decision-ready cohort in the window", {"payers": 0, **pop}, evaluable=False)
    g = np.sort(np.asarray(gross_by_payer, dtype=float))[::-1]
    if len(g) == 0:
        return CheckResult("revenue_concentration", True, "no IAP revenue among eligible cohorts in window", {"payers": 0, **pop}, evaluable=False)
    k = max(1, math.ceil(0.01 * len(g)))
    share = float(g[:k].sum() / g.sum())
    ok = share <= top_share_limit
    partial = bool(pop and pop["excluded_dates"])
    return CheckResult("revenue_concentration", bool(ok),
                       f"top {k} payer(s) of {len(g)} provide {share:.0%} of IAP" + ("" if ok else f" (limit {top_share_limit:.0%})")
                       + (f"; computed over {len(eligible)} of {len(window_dates)} window dates (incomplete or missing dates excluded)" if partial else ""),
                       {"payers": int(len(g)), "top_k": k, "top_share": share, "limit": top_share_limit, **pop})


def check_cpi_anomaly(window_cpi, trailing_median, high, low) -> CheckResult:
    if window_cpi is None or trailing_median is None or np.isnan(window_cpi) or np.isnan(trailing_median) or trailing_median <= 0:
        return CheckResult("cpi_anomaly", True, "not evaluable: no CPI baseline", {"window_cpi": _num(window_cpi), "trailing_median": _num(trailing_median)}, evaluable=False)
    ratio = window_cpi / trailing_median
    ok = low <= ratio <= high
    return CheckResult("cpi_anomaly", bool(ok), f"window CPI is {ratio:.2f} x the trailing median" + ("" if ok else f" (allowed {low}..{high})"),
                       {"window_cpi": float(window_cpi), "trailing_median": float(trailing_median), "ratio": float(ratio)})


def check_tracking_gap(daily: pd.DataFrame, declared_lag_days: int, as_of_date: str, q: dict) -> CheckResult:
    """Events per recently installed player collapsing on the same calendar days signals missing tracking, not poor players.
    `daily` is daily_activity output. Days whose events may still be unreported (last declared_lag_days + 1 days before as_of)
    are ignored, so reporting delay is not mistaken for a gap.
    Design (revised after the first dev backtest, see reports/phase2_report.md): the baseline is the MEDIAN activity per player
    over the `tracking_gap_baseline_days` complete days before the recent window, so an outage that straddles the recent and
    baseline windows cannot hide itself; a gap is suspected when at least `tracking_gap_min_run_days` CONSECUTIVE recent days
    fall below (1 - tracking_gap_drop) x that median."""
    if daily.empty:
        return CheckResult("tracking_gap", True, "not evaluable: no activity data", {}, evaluable=False)
    last_complete = pd.Timestamp(as_of_date) - pd.Timedelta(days=declared_lag_days + 1)
    d = daily[pd.to_datetime(daily.event_date) <= last_complete].copy()
    d = d[d.recent_installs > 0]
    d["per_player"] = d.active_player_days / d.recent_installs
    r, b = q["tracking_gap_recent_days"], q["tracking_gap_baseline_days"]
    if len(d) < r + b:
        return CheckResult("tracking_gap", True, "not evaluable: fewer than recent+baseline days of complete data", {"days_available": int(len(d))}, evaluable=False)
    recent = d.tail(r)
    base = float(d.iloc[-(r + b):-r].per_player.median())
    floor = (1 - q["tracking_gap_drop"]) * base
    below = (recent.per_player < floor).tolist()
    run = best = 0
    for flag in below:
        run = run + 1 if flag else 0
        best = max(best, run)
    ok = best < q["tracking_gap_min_run_days"]
    return CheckResult("tracking_gap", ok,
                       f"longest run of recent days below {1 - q['tracking_gap_drop']:.0%} of the prior {b}-day median activity per player is {best} day(s)"
                       + ("" if ok else f" (limit {q['tracking_gap_min_run_days'] - 1}): suspected tracking gap, not player quality"),
                       {"baseline_median_per_player": base, "floor": float(floor), "longest_run_days": best,
                        "days_below_floor": recent.event_date[recent.per_player < floor].tolist(),
                        "recent_days": [recent.event_date.iloc[0], recent.event_date.iloc[-1]]})


def check_window_cpi(cohorts: pd.DataFrame, window_dates, q: dict) -> CheckResult:
    """Aggregate CPI of the window = total window spend / total window installs, over every DECISION-READY cohort of one channel
    with known spend (not only cohorts that received a forecast, so excluding small or odd cohorts cannot hide a CPI move).
    A complete cohort with spend and ZERO installs contributes its spend and zero installs (a known zero), so it raises the CPI;
    it is not dropped. Cohorts whose install count is not yet final (D3 incomplete) are UNKNOWN, not zero: they are excluded and
    listed in the evidence. Compared with the median per-cohort CPI of the previous `cpi_baseline_days` decision-ready cohorts
    that had at least one install. `cohorts` = annotate_cohorts output for one channel."""
    dates = list(window_dates)
    inwin = cohorts[cohorts.install_date.isin(dates)]
    complete = inwin[inwin.d3_complete == 1]
    known = complete[complete.spend.notna()]
    ev = {"cohorts_in_window": int(len(known)),
          "excluded_incomplete_dates": sorted(inwin[inwin.d3_complete != 1].install_date),
          "excluded_missing_spend_dates": sorted(complete[complete.spend.isna()].install_date)}
    spend, installs = float(known.spend.sum()), int(known.installs_reported.sum())
    if len(known) == 0 or (spend == 0 and installs == 0):
        return CheckResult("cpi_anomaly", True, "not evaluable: no decision-ready cohort with known spend in the window", {**ev, "spend": spend, "installs": installs}, evaluable=False)
    if installs == 0:
        return CheckResult("cpi_anomaly", False, f"unusable: spend {spend:.2f} bought zero installs in the window (CPI undefined/infinite), not a normal window",
                           {**ev, "spend": spend, "installs": 0})
    start = min(dates)
    prior = cohorts[(cohorts.install_date < start) & (cohorts.d3_complete == 1) & cohorts.cpi.notna()].tail(q["cpi_baseline_days"])
    median = float(prior.cpi.median()) if len(prior) >= 5 else float("nan")
    res = check_cpi_anomaly(spend / installs, median, q["cpi_anomaly_high"], q["cpi_anomaly_low"])
    res.evidence.update(ev, spend=spend, installs=installs)
    return res
