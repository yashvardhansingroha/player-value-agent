"""Generate reports/phase2_report.md from the stored backtest results and forecast store. Every number comes from data.

    python scripts/phase2_report.py [--seed 1]
(run scripts/run_backtest.py first)
"""
import argparse
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecast import store  # noqa: E402
from src.forecast.pipeline import run_review  # noqa: E402
from tests.test_forecast_toy import COHORTS, REVIEW_DAY, SCORING_DAY, D, build_toy, toy_cfg  # noqa: E402
from src.forecast import scoring  # noqa: E402


def md(df, floats=3):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if v is None or (isinstance(v, float) and pd.isna(v)) else (f"{v:.{floats}f}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(lines)


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def metric_row(name, m):
    return {"slice": name, "scored cohorts": m.get("n", 0), "WAPE": pct(m.get("wape")), "bias": pct(m.get("bias")),
            "with interval": m.get("n_with_interval", 0), "coverage (80% nominal)": pct(m.get("coverage")),
            "median width / forecast": "" if "interval_width_over_forecast" not in m else f"{m['interval_width_over_forecast']:.2f}"}


def toy_section():
    tmp = Path(tempfile.mkdtemp())
    build_toy(tmp / "toy.db")
    dec = sqlite3.connect(tmp / "toy.db")
    st = store.create_store(tmp / "toy_f.db")
    run_review(dec, st, D(REVIEW_DAY), toy_cfg())
    f = pd.read_sql_query("SELECT * FROM forecast_issued", st)
    sc = scoring.score_cohorts(st, dec, D(SCORING_DAY))["overall"]
    L = ["### A. Hand-checkable toy example (the same case is asserted in `tests/test_forecast_toy.py`)", "",
         "Five cohorts of one channel, 2 players each. Review on day 34; D30 outcomes of c0-c2 are already complete, c3 and c4 are newly decision-ready.", "",
         md(pd.DataFrame([{"cohort": n, "install day": d, "D0-D3 net (y3)": y3, "D0-D30 net (y30)": y3 + e} for n, d, y3, e in COHORTS]), 1), "",
         "- training (c0..c2): sum(y3) = 10 + 20 + 10 = 40; sum(y30) = 30 + 50 + 40 = 120; multiplier = 120 / 40 = **3.0** (ratio of sums; the mean of the three ratios 3, 2.5, 4 would be 3.17).",
         f"- c3 forecast = 3.0 x 15 = **{f[f.install_date == D(29)].point_forecast.iloc[0]:.1f}** (pipeline output). Reference ignoring D3 = (120 / 6 installs) x 2 = {f[f.install_date == D(29)].reference_forecast.iloc[0]:.1f}.",
         f"- c4 has zero activity (D0-D3 net = 0): forecast uses the fallback, **{f[f.install_date == D(28)].point_forecast.iloc[0]:.1f}** (not 3.0 x 0 = 0), method `{f[f.install_date == D(28)].method.iloc[0]}`.",
         f"- scored on day 70 (outcomes complete): sum(actual) = {sc['sum_actual']:.0f}, sum(forecast) = {sc['sum_forecast']:.0f}; WAPE = |60-45| + |0-40| over 60 = {sc['wape']:.4f}; bias = (85-60)/60 = {sc['bias']:.4f}.", ""]
    dec.close()
    st.close()
    return "\n".join(L)


def real_sample_section(st, dec):
    """Pick one issued channel-multiplier forecast, recompute its inputs from RAW tables and check the stored numbers."""
    issued = pd.read_sql_query("SELECT * FROM forecast_issued WHERE method = 'channel_multiplier' AND interval_status = 'available' AND channel = 'meta' ORDER BY issued_as_of", st)
    r = issued.iloc[len(issued) // 2]
    tr = json.loads(r.training_json)
    ft = json.loads(r.features_json)
    held = pd.read_sql_query("SELECT install_date, reason FROM forecast_withheld WHERE channel = ?", st, params=(r.channel,))
    iss = pd.read_sql_query("SELECT install_date, method FROM forecast_issued WHERE channel = ?", st, params=(r.channel,))
    quality_reasons = {"missing_spend", "no_installs", "below_min_installs", "low_activity", "revenue_spike"}
    excluded = set(held[held.reason.isin(quality_reasons)].install_date) | set(iss[iss.method == "zero_d3_fallback"].install_date)
    cohorts = pd.read_sql_query(
        "SELECT p.install_date, COUNT(DISTINCT p.player_id) AS installs, "
        "SUM(CASE WHEN d.day_n <= 3 THEN d.iap_gross * 0.7 + d.ad_revenue ELSE 0 END) AS y3, "
        "SUM(CASE WHEN d.day_n <= 30 THEN d.iap_gross * 0.7 + d.ad_revenue ELSE 0 END) AS y30 "
        "FROM players p LEFT JOIN player_days d ON d.player_id = p.player_id WHERE p.channel = ? AND p.install_date BETWEEN ? AND ? GROUP BY p.install_date",
        dec, params=(r.channel, tr["first_install"], tr["last_install"]))
    train = cohorts[~cohorts.install_date.isin(excluded)]
    ok = (len(train) == tr["n_cohorts"] and abs(train.y3.sum() - tr["sum_y3"]) < 1e-6 and abs(train.y30.sum() - tr["sum_y30"]) < 1e-6)
    cohort_actual = dec.execute(
        "SELECT SUM(d.iap_gross) * 0.7 + SUM(d.ad_revenue) FROM player_days d JOIN players p ON p.player_id = d.player_id "
        "WHERE p.channel = ? AND p.install_date = ? AND d.day_n <= 30", (r.channel, r.install_date)).fetchone()[0]
    L = ["### B. One real dev-seed forecast, recomputed from raw tables", "",
         f"Cohort `{r.channel}` installed {r.install_date}, issued at review {r.issued_as_of}, method `{r.method}`.", "",
         f"- stored D0-D3 net revenue (feature): {ft['d0_3_net_revenue']:.4f} from {ft['installs_reported']} installs",
         f"- stored training window: {tr['n_cohorts']} eligible D30-complete cohorts installed {tr['first_install']} .. {tr['last_install']}; latest training outcome complete on {r.train_max_outcome_complete_at} (<= issue date {r.issued_as_of})",
         f"- stored sums: sum(y3) = {tr['sum_y3']:.3f}, sum(y30) = {tr['sum_y30']:.3f}; multiplier = {tr['sum_y30']:.3f} / {tr['sum_y3']:.3f} = {tr['sum_y30'] / tr['sum_y3']:.4f} (stored {tr['multiplier']:.4f})",
         f"- forecast = {tr['multiplier']:.4f} x {ft['d0_3_net_revenue']:.4f} = {tr['multiplier'] * ft['d0_3_net_revenue']:.3f} (stored {r.point_forecast:.3f})",
         f"- 80% interval from {r.calib_n} earlier matured forecasts (latest outcome complete {r.calib_max_outcome_complete_at}): [{r.interval_lo:.3f}, {r.interval_hi:.3f}]",
         f"- later actual D0-D30 net revenue: {cohort_actual:.3f} ({'inside' if r.interval_lo <= cohort_actual <= r.interval_hi else 'outside'} the interval)",
         f"- **independent recomputation of the training sums from the raw tables** (cohorts in that date range minus those withheld/flagged for quality): "
         f"{len(train)} cohorts, sum(y3) = {train.y3.sum():.3f}, sum(y30) = {train.y30.sum():.3f} -> **{'MATCH' if ok else 'MISMATCH'}**", ""]
    return "\n".join(L)


def flag_table(store_path):
    """(channel, as_of, check) sets that FAILED (evaluable) and NOT EVALUABLE, from a store's weekly summaries."""
    c = sqlite3.connect(store_path)
    failed, noteval = set(), set()
    for ch, as_of, qj in c.execute("SELECT channel, as_of, quality_json FROM weekly_summary"):
        for x in json.loads(qj):
            if not x["evaluable"]:
                noteval.add((ch, as_of, x["name"]))
            elif not x["passed"]:
                failed.add((ch, as_of, x["name"]))
    c.close()
    return failed, noteval


def accounting_section(st, c, s):
    summ = pd.read_sql_query("SELECT * FROM weekly_summary", st)
    status = {}
    for r in summ.itertuples(index=False):
        for a in json.loads(r.accounting_json):
            status[a["status"]] = status.get(a["status"], 0) + 1
    not_ready = summ[summ.fully_decision_ready == 0]
    reasons = {}
    for r in not_ready.itertuples(index=False):
        for x in json.loads(r.not_ready_reasons_json):
            key = x.split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1
    f = pd.read_sql_query("SELECT * FROM forecast_issued", st)
    L = ["## 5. Window accounting, readiness and the cumulative-revenue floor", "",
         "Every expected install date in each weekly window is accounted for as `forecast`, `withheld`, `pending_incomplete` (cohort exists but D3 is not declared complete: "
         "install count unknown, not zero), `missing` (no cohort data visible at all) or `unaccounted` (a complete cohort with no stored record: should never occur). "
         "A complete cohort with zero activity is a known zero (it is forecast or withheld, never `missing`). Dates before a channel's first visible data are not expected.", "",
         md(pd.DataFrame([{"status": k, "window dates (all summaries)": v} for k, v in sorted(status.items())])), "",
         f"- summaries written: {len(summ)}; fully decision-ready: {int((summ.fully_decision_ready == 1).sum())}; not fully decision-ready: {len(not_ready)}"
         + (f" (reasons: {reasons})" if reasons else " (none on this dev seed: no interior incomplete or missing dates occurred, so the blocking path is exercised by the regression tests, not by dev data)"),
         f"- scored summaries by readiness: fully ready {s['fully_decision_ready'].get('n', 0)}, not fully ready {s['not_fully_decision_ready'].get('n', 0)}", "",
         "**Cumulative-revenue floor.** D30 cumulative net revenue cannot be below the D0-D3 net revenue already observed. Point forecasts and interval bounds (cohort and weekly summary) "
         "are floored at it, keeping lo <= hi. This is a structural consistency correction: it only raises bounds that were below the floor (it never widens an interval) and is not a coverage fix.", "",
         f"- cohort forecasts: point forecasts raised to the floor: {c['floor_adjustments']['point_forecasts_raised']} of {c['issued']}; intervals adjusted: {c['floor_adjustments']['intervals_adjusted']} of "
         f"{int((f.interval_status == 'available').sum())}; actual below the observed D3 floor: {c['floor_adjustments']['actual_below_observed_d3_floor']}; unordered intervals: {c['floor_adjustments']['any_interval_unordered']}",
         f"- weekly summaries: intervals adjusted: {s['floor_adjustments']['intervals_adjusted']}; actual below the floor: {s['floor_adjustments']['actual_below_observed_d3_floor']}; unordered intervals: {s['floor_adjustments']['any_interval_unordered']}", ""]
    return L


def provenance_and_changes_section(st, res):
    prev_dir = ROOT / "data" / "scratch" / "prev"
    L = ["## 6. Provenance, amendment history, and what changed since 7bafc44", "",
         "Amendment history (kept, not rewritten):", "",
         md(pd.DataFrame([
             {"when": "first dev backtest", "what changed": "`tracking_gap`: algorithm (mean-vs-mean over 14 days -> median baseline over 28 days + run of 3 consecutive low days); `cpi_anomaly`: population (forecast cohorts only -> all decision-ready cohorts)",
              "why": "neither fired on the dev outage / CPI spike"},
             {"when": "independent review of 7bafc44", "what changed": "weekly windows now account for every expected date; completeness is checked inside the window; payer-concentration SQL restricted to D3-complete cohorts; `cpi_anomaly` keeps zero-install spend; cumulative-revenue floor added",
              "why": "an interior incomplete date vanished from both lists; zero-install spend was dropped from aggregate CPI; D30 intervals could sit below observed D3 revenue"}])), "",
         "Settings that stayed fixed: all `forecast:` and `backtest:` settings, `min_installs_per_cohort`, `baseline_window_cohorts`, `baseline_min_cohorts`, `spike_ratio`, `low_activity_ratio`, "
         "`window_min_installs`, `concentration_top_share`, `cpi_baseline_days`, `cpi_anomaly_high/low`, `tracking_gap_recent_days`, `tracking_gap_drop`. "
         "Thresholds were not changed to improve a score.", ""]
    if not (prev_dir / "results_7bafc44.json").exists():
        return L + ["(previous-run snapshot not available in this checkout; comparison omitted)", ""]
    prev = json.loads((prev_dir / "results_7bafc44.json").read_text())
    rows = []
    for name, p, n in (("cohort forecasts: all scored", prev["cohort_forecasts"]["overall"], res["cohort_forecasts"]["overall"]),
                       ("weekly summaries: all scored", prev["weekly_summaries"]["overall"], res["weekly_summaries"]["overall"])):
        for label, pm in (("7bafc44", p), ("now", n)):
            rows.append({"slice": name, "version": label, "n": pm.get("n"), "WAPE": pct(pm.get("wape")), "bias": pct(pm.get("bias")),
                         "with interval": pm.get("n_with_interval"), "coverage": pct(pm.get("coverage")),
                         "median width / forecast": "" if "interval_width_over_forecast" not in pm else f"{pm['interval_width_over_forecast']:.3f}"})
    L += ["Scores before (7bafc44) and now:", "", md(pd.DataFrame(rows)), ""]
    same_points = prev["cohort_forecasts"]["overall"]["wape"] == res["cohort_forecasts"]["overall"]["wape"] and prev["cohort_forecasts"]["overall"]["bias"] == res["cohort_forecasts"]["overall"]["bias"]
    L += [f"Cohort point forecasts and WAPE/bias identical to 7bafc44: {'yes' if same_points else 'NO'}.", ""]
    pf, pn = flag_table(prev_dir / "forecasts_7bafc44.db")
    nf, nn = flag_table(ROOT / "data" / "forecasts" / f"seed_{res['seed']}" / "forecasts.db")
    checks_all = sorted({k[2] for k in pf | nf | pn | nn})
    rows = []
    for ck in checks_all:
        pfc, nfc = {k[:2] for k in pf if k[2] == ck}, {k[:2] for k in nf if k[2] == ck}
        pnc, nnc = {k[:2] for k in pn if k[2] == ck}, {k[:2] for k in nn if k[2] == ck}
        ex = lambda s_: ", ".join(f"{a}@{b}" for a, b in sorted(s_)[:4]) + (" ..." if len(s_) > 4 else "")
        rows.append({"check": ck, "failed before": len(pfc), "failed now": len(nfc), "newly failed": len(nfc - pfc), "no longer failed": len(pfc - nfc),
                     "not evaluable before": len(pnc), "now": len(nnc), "examples (newly failed / no longer failed)": f"{ex(nfc - pfc)} / {ex(pfc - nfc)}"})
    L += ["Window-check flags before and now (failed = evaluable and failed):", "", md(pd.DataFrame(rows)), "",
          "Why flags can differ: `missing_days` no longer flags dates before a channel's first data; `cpi_anomaly` now keeps zero-install spend and reports excluded incomplete dates; "
          "`revenue_concentration` now uses D3-complete cohorts only; `reporting_completeness` now also fails on incomplete dates INSIDE the window. Flag changes do not alter which cohorts are forecast.", ""]
    return L


def main(seed):
    res = json.loads((ROOT / "reports" / "phase2_backtest_results.json").read_text())
    st = sqlite3.connect(ROOT / "data" / "forecasts" / f"seed_{seed}" / "forecasts.db")
    dec = sqlite3.connect(ROOT / "data" / "decision" / f"seed_{seed}" / "decision.db")
    c, s = res["cohort_forecasts"], res["weekly_summaries"]
    cfg = res["config"]
    L = ["# Phase 2 report (generated by scripts/phase2_report.py)", "",
         "ALL DATA IS SYNTHETIC. DEV SEED ONLY: the generator was calibrated on this seed, so these scores are development results, "
         "not held-out evidence.", "",
         "**Configuration provenance (precise).** The forecast settings (training window, minimum training cohorts, pooling thresholds, interval alpha, "
         "calibration windows and minimums, summary length), the backtest settings, and the cohort-level quality thresholds (minimum installs, spike ratio, "
         "low-activity ratio, baseline windows) were fixed before any backtest score was seen and have not changed. Several DETECTOR algorithms and settings "
         "(tracking gap, CPI-anomaly population, window completeness accounting, payer-concentration population) were changed after development results or "
         "independent review were inspected; the full amendment history is in the header of `config/forecast.yaml` and in section 5 below. "
         "Because detectors were adjusted after seeing the dev data, dev flags are not held-out evidence. Observed interval coverage is below the 80% "
         "nominal level; coverage is measured, never guaranteed.", "",
         "## 1. Hand-checkable baseline calculations", "", toy_section(), real_sample_section(st, dec)]

    first, last = c["ranges"]["issued_as_of"], c["ranges"]["install_date"]
    L += ["## 2. Backtest: cohort forecasts", "",
          f"- reviews: Mondays, day {cfg['backtest']['first_as_of_day']}..{cfg['backtest']['last_as_of_day']} ({len(res['reviews'])} reviews); backtest runtime {res['backtest_seconds']} s",
          f"- issued forecasts: {c['issued']}; scored (outcome matured at scoring date {c['scoring_as_of']}): {c['scored']}; pending (outcome not in storage): {c['pending_outcome_not_in_storage']}",
          f"- scored install dates: {last[0]} .. {last[1]}; scored issue dates (reviews): {first[0]} .. {first[1]}",
          f"- WAPE = sum|actual-forecast| / sum(actual); bias = sum(forecast-actual) / sum(actual) (positive = over-forecast); coverage = share of actuals inside the 80% interval, among forecasts that have one. Total-revenue WAPE: install weights cancel, so there is no second 'install-weighted' WAPE.", ""]
    rows = [metric_row("ALL scored cohorts", c["overall"]), metric_row("only cohorts with an interval", c["overall_with_interval_only"])]
    rows += [metric_row(f"method: {k}", v) for k, v in c["by_method"].items()]
    rows += [metric_row(f"channel: {k}", v) for k, v in c["by_channel"].items()]
    rows += [metric_row(f"period {k}", v) for k, v in c["by_period"].items()]
    L += [md(pd.DataFrame(rows)), ""]
    ref = c["reference_trailing_value_per_install_ignoring_d3"]
    L += [f"Reference forecast that ignores D3 revenue (training average D30 value per install x installs), same cohorts: WAPE {pct(ref['wape'])}, bias {pct(ref['bias'])}.", "",
          "Coverage by period, only cohorts with an interval (calibration comes from EARLIER matured forecasts; coverage is measured on the later cohort):", "",
          md(pd.DataFrame([metric_row(f"period {k}", v) for k, v in c["by_period_with_interval"].items()] +
                          [metric_row(f"channel {k}", v) for k, v in c["by_channel_with_interval"].items()])), "",
          f"Intervals unavailable for {sum(c['interval_unavailable_reasons'].values())} issued forecasts: {c['interval_unavailable_reasons']}.", ""]

    L += ["## 3. Backtest: trailing-seven-day channel summaries (separate from cohort forecasts)", "",
          "Each summary covers the 7 consecutive install dates ending at the channel's latest decision-ready install date at that review "
          "(for example, a channel with a long reporting lag ends earlier). It sums the stored forecasts of the cohorts that were ISSUED in that window; "
          "withheld cohorts are listed with the summary and excluded from both forecast and actual. Intervals are calibrated on earlier matured summaries.", "",
          f"- summaries written: {s['summaries']}; scored: {s['scored']}; empty (no included cohort): {s['empty_no_included_cohorts']}; pending: {s['pending_not_mature']}",
          f"- scored review dates {s['ranges']['as_of'][0]} .. {s['ranges']['as_of'][1]}; summary windows {s['ranges']['window_start'][0]} .. {s['ranges']['window_end'][1]}", "",
          md(pd.DataFrame([metric_row("ALL scored summaries", s["overall"])] + [metric_row(f"channel: {k}", v) for k, v in s["by_channel"].items()] +
                          [metric_row("summaries with a failed quality check", s["flagged_by_quality_check"]), metric_row("summaries with no failed check", s["unflagged"])])), ""]

    L += ["## 4. Quality-check outcomes and exclusions", "",
          f"Cohorts attempted: {c['issued'] + c['withheld_total']} (issued {c['issued']}, withheld {c['withheld_total']}). Withheld cohorts carry "
          f"{pct(c['share_of_matured_d30_revenue_withheld'])} of the matured D30 net revenue of all attempted cohorts, so scored accuracy above describes the issued cohorts only.", ""]
    wrows = [{"reason": k, "cohorts": v["cohorts"], "matured D30 net revenue they carried": f"{v['matured_actual_d30_net_revenue']:.0f}", "by channel": v["by_channel"]} for k, v in c["withheld"].items()]
    L += [md(pd.DataFrame(wrows)), ""]
    sens = res["sensitivity_no_spike_exclusion"]
    L += ["### Sensitivity (NOT the chosen configuration): revenue-spike exclusion switched off", "",
          md(pd.DataFrame([metric_row("all cohorts, no spike exclusion", sens["all_cohorts"]),
                           metric_row("cohorts the main run withheld for a spike, forecast anyway", sens["on_cohorts_the_main_run_withheld_for_spike"]),
                           metric_row("cohorts the main run issued, under the no-exclusion run", sens["on_cohorts_the_main_run_issued"])])), "",
          "The spike exclusion removes the highest-revenue cohorts; the row for those cohorts shows how the baseline would have done on them.", ""]

    summ = pd.read_sql_query("SELECT channel, as_of, quality_json FROM weekly_summary", st)
    flags = []
    for r in summ.itertuples(index=False):
        for ch in json.loads(r.quality_json):
            if ch["evaluable"] and not ch["passed"]:
                flags.append({"channel": r.channel, "as_of": r.as_of, "check": ch["name"], "reason": ch["reason"]})
            elif not ch["evaluable"]:
                flags.append({"channel": r.channel, "as_of": r.as_of, "check": ch["name"], "reason": "NOT EVALUABLE"})
    fl = pd.DataFrame(flags)
    L += ["### Window-level checks across all channel summaries", "",
          md(fl.groupby(["check", fl.reason.where(fl.reason == "NOT EVALUABLE", "FAILED")]).size().reset_index(name="summaries").rename(columns={"reason": "result"})), ""]
    for check in ("tracking_gap", "cpi_anomaly", "revenue_concentration", "window_min_installs", "missing_days"):
        sub = fl[(fl.check == check) & (fl.reason != "NOT EVALUABLE")]
        if len(sub):
            L += [f"`{check}` failed on: " + "; ".join(f"{ch}: {', '.join(g.as_of)}" for ch, g in sub.groupby("channel")), ""]
    L += ["`reporting_completeness` is informational: it fails only when more install dates are pending than the declared lag explains.", ""]

    early = pd.read_sql_query("SELECT attempted_as_of, COUNT(*) AS n FROM forecast_withheld WHERE reason = 'insufficient_training_history' GROUP BY attempted_as_of", st)
    L += ["### Explicit limitations reported by the pipeline", "",
          f"- insufficient training history: {int(early.n.sum())} cohorts withheld at reviews {early.attempted_as_of.min()} .. {early.attempted_as_of.max()} (need {cfg['forecast']['min_train_cohorts']} eligible D30-complete cohorts in the channel, or {cfg['forecast']['min_pooled_cohorts']} pooled).",
          f"- interval unavailable (not enough matured earlier forecasts, need {cfg['forecast']['cohort_min_calibration']} per channel): {c['interval_unavailable_reasons']}.", ""]
    L += accounting_section(st, c, s)
    L += provenance_and_changes_section(st, res)
    out = ROOT / "reports" / "phase2_report.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    main(p.parse_args().seed)
