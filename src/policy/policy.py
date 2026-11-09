"""Deterministic budget policy for one Monday review. Every number comes from Python/SQL; no LLM, no hidden truth, no outcome
from after the decision date (decision-date inputs: the forecast store's records for THIS review, and availability-filtered data).

STAGES (evaluated separately and all recorded in the decision):
  0 evidence binding   the weekly summary and every forecast it sums must carry the hash of the forecast/quality configuration this
                       review runs under, and the summary must equal the sum of its contributing forecasts. Otherwise the decision is
                       insufficient_evidence and the evidence must be regenerated under the current configuration (never reused).
  1 completeness       weekly summary `fully_decision_ready` = every window date accounted for (NOT "all checks passed").
  2 quality checks     each window check is classified by what its failure blocks (table QUALITY_BLOCKS below).
  3 forecast coverage  the weekly forecast covers only its INCLUDED cohorts. Their share of the window's decision-ready spend and
                       their count must be adequate; excluded cohorts are reported from decision-date data (never their later D30).
  4 uncertainty        the summary interval must exist and be usable (not too wide).
  5 economics          a payback target must be derivable from mature history; increases need a usable response curve and a step
                       inside the observed spend range.

WHAT BLOCKS WHAT
  insufficient_evidence (nothing can be recommended): window not fully decision-ready; tracking_gap, window_min_installs,
      missing_days or reporting_completeness failed; too few / too little-spend included cohorts; no usable interval; no payback
      target; an increase that needs a response curve the data cannot support (unusable curve, outside the observed range).
  hold (evidence adequate, no change justified): cpi_anomaly failed (a transient price move is not a response curve), the rules
      below not met, or a constraint (reserve, share) leaves no room.
  increases blocked only: revenue_concentration failed, tracking_gap / cpi_anomaly not evaluable.
  decreases blocked only: included cohorts exclude spike-withheld cohorts (their revenue is higher, so the included-only average
      ROAS is biased low).

RULES (after the gates) are DOCUMENTED POLICY HEURISTICS, not proofs of economic safety
  decrease       : upper bound of the forecast average D30 ROAS of the included cohorts < target D30 ROAS.
                   Heuristic assumptions: the included cohorts represent the channel's recent players; the forecast and its interval
                   are adequate (observed summary-interval coverage on the dev seed is below nominal); and spend at the margin is
                   not worth MORE than the average. The last one is NOT guaranteed: marginal ROAS can exceed average ROAS for some
                   response shapes (the configuration permits an elasticity above one) or if player value changes with spend.
                   A decrease is a bounded step (at most the configured step, floor-limited), reviewed by a human, and reversible.
  increase_test  : average ROAS point >= target AND lower bound of the MODEL-ESTIMATED finite-step marginal D30 ROAS >= target.
                   Requiring average ROAS >= target first is the same diminishing-returns heuristic; it is not exact for every curve.
  otherwise hold.
An increase_test is a small controlled TEST that still needs a pre-registered experiment plan (Phase 4): it is proposed with
qualification `plan_required` and the ledger refuses to apply it without a registered frozen plan.
"""
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.metrics import cohort_metrics
from src.forecast import store as fstore
from src.ledger.proposal import ProposalItem, Proposal, jsonable, make_proposal_id
from src.policy import constraints as cons
from src.policy import economics as eco
from src.quality import checks as qchecks

Z80 = 1.2815515655446004  # an 80% central interval spans +/- 1.28 standard deviations

QUALITY_BLOCKS = {  # check name -> (what a failure blocks, resulting outcome)
    "reporting_completeness": ("all", "insufficient_evidence"),
    "missing_days": ("all", "insufficient_evidence"),
    "window_min_installs": ("all", "insufficient_evidence"),
    "tracking_gap": ("all", "insufficient_evidence"),
    "cpi_anomaly": ("all", "hold"),
    "revenue_concentration": ("increase", None),
}
NOT_EVALUABLE_BLOCKS_INCREASE = ("tracking_gap", "cpi_anomaly")  # cannot verify data health or price stability


@dataclass
class Review:
    as_of: str
    dec: object
    st: object
    pcfg: dict
    fcfg: dict
    m: pd.DataFrame
    ann: pd.DataFrame
    mature: pd.DataFrame


def build_review(dec, st, as_of: str, pcfg: dict = None, fcfg: dict = None) -> Review:
    pcfg = pcfg or cons.load_policy()
    fcfg = fcfg or cons.load_forecast_config()
    m = cohort_metrics(dec, as_of)
    ann = qchecks.annotate_cohorts(m[m.channel != "organic"], fcfg["quality"])
    return Review(as_of, dec, st, pcfg, fcfg, m, ann, eco.load_mature_history(dec, as_of, ann))


def load_summary(st, channel: str, as_of: str):
    df = pd.read_sql_query("SELECT * FROM weekly_summary WHERE channel = ? AND as_of = ?", st, params=(channel, as_of))
    return None if df.empty else df.iloc[0].to_dict()


def verify_forecast_evidence(st, S: dict, fcfg: dict) -> dict:
    """Is this weekly summary usable under the forecast/quality configuration of the current review?
    1. the summary's stored config hash must equal the hash of `fcfg` (else: stale or foreign evidence);
    2. every cohort it lists as forecast must exist in forecast_issued with the same hash;
    3. the summary's point forecast must equal the sum of those contributing forecasts.
    Returns {"ok": True} or {"ok": False, "code": ..., ...evidence for the reader}."""
    expected = fstore.forecast_evidence_hash(fcfg)
    if S["config_hash"] != expected:
        return {"ok": False, "code": "forecast_evidence_config_mismatch", "stored_hash": S["config_hash"], "expected_hash": expected}
    dates = [a["install_date"] for a in json.loads(S["accounting_json"]) if a["status"] == "forecast"]
    rows = pd.read_sql_query("SELECT install_date, point_forecast, config_hash FROM forecast_issued WHERE channel = ? AND install_date BETWEEN ? AND ?",
                             st, params=(S["channel"], S["window_start"], S["window_end"]))
    rows = rows[rows.install_date.isin(dates)]
    missing = sorted(set(dates) - set(rows.install_date))
    foreign = sorted(rows[rows.config_hash != expected].install_date)
    total = float(rows.point_forecast.sum()) if len(rows) else 0.0
    summary_point = 0.0 if pd.isna(S["point_forecast"]) else float(S["point_forecast"])
    sum_ok = abs(total - summary_point) <= 1e-6 * max(1.0, abs(summary_point))
    if missing or foreign or not sum_ok:
        return {"ok": False, "code": "forecast_evidence_inconsistent", "expected_hash": expected, "summary_hash": S["config_hash"],
                "contributing_forecasts_missing": missing, "contributing_forecasts_with_other_config": foreign,
                "summary_point_forecast": summary_point, "sum_of_contributing_forecasts": total}
    meta = fstore.read_meta(st)  # informational: the run schedule that produced the store is recorded separately from the evidence hash
    return {"ok": True, "expected_hash": expected, "contributing_forecasts": len(dates),
            "run_schedule": {"schedule_hash": meta.get("schedule_hash"), "schedule": json.loads(meta["schedule_json"]) if "schedule_json" in meta else None,
                             "note": "scope of the evidence hash: forecasting and quality settings only; run scheduling is recorded here separately"}}


def initial_budgets_from_spend(dec, as_of: str, days: int = 28) -> dict:
    """Sandbox starting budgets: each paid channel's mean visible daily spend over its last `days` dates (a decision-date number)."""
    m = cohort_metrics(dec, as_of)
    m = m[(m.channel != "organic") & m.spend.notna()].sort_values("install_date")
    return {ch: round(float(g.tail(days).spend.mean()), 1) for ch, g in m.groupby("channel")}


def populations(rv: Review, ch: str, S: dict) -> dict:
    """Included and excluded populations from decision-date information only. Excluded cohorts are described by what is known at the
    decision date (installs, spend, observed D0-D3 revenue, reason); their later D30 revenue is never used."""
    acc = json.loads(S["accounting_json"])
    rows = rv.ann[rv.ann.channel == ch].set_index("install_date")
    included = [a["install_date"] for a in acc if a["status"] == "forecast"]
    excluded = []
    for a in acc:
        if a["status"] == "withheld":
            r = rows.loc[a["install_date"]]
            excluded.append({"install_date": a["install_date"], "reason": a["reason"], "installs": int(r.installs_reported),
                             "spend": None if pd.isna(r.spend) else float(r.spend), "observed_d0_3_net_revenue": None if pd.isna(r.d0_3_net_revenue) else float(r.d0_3_net_revenue)})
    inc_spend, inc_installs = float(S["spend_included"]), int(S["installs_included"])
    exc_spend = float(sum(e["spend"] or 0.0 for e in excluded))
    inc_d3 = float(sum(rows.loc[d].d0_3_net_revenue for d in included)) if included else 0.0
    return {
        "note": "The weekly forecast covers ONLY the included cohorts; it is not a whole-channel forecast.",
        "n_expected_dates": int(S["n_expected_dates"]), "window": [S["window_start"], S["window_end"]],
        "included": {"install_dates": included, "n_cohorts": len(included), "installs": inc_installs, "spend": inc_spend,
                     "forecast_d30_net_revenue": None if pd.isna(S["point_forecast"]) else float(S["point_forecast"]),
                     "observed_d0_3_net_revenue": inc_d3},
        "excluded_withheld": {"cohorts": excluded, "n_cohorts": len(excluded), "installs": int(sum(e["installs"] for e in excluded)), "spend": exc_spend,
                              "observed_d0_3_net_revenue": float(sum(e["observed_d0_3_net_revenue"] or 0.0 for e in excluded))},
        "included_spend_share": inc_spend / (inc_spend + exc_spend) if (inc_spend + exc_spend) > 0 else None,
        "accounting_statuses": {k: sum(1 for a in acc if a["status"] == k) for k in sorted({a["status"] for a in acc})},
    }


def _stage(d, name, passed, detail, blocks="none"):
    d["stages"].append({"stage": name, "passed": bool(passed), "blocks": blocks, "detail": detail})


def decide_channel(rv: Review, ch: str, budget: float, ceiling: float, budgets: dict) -> dict:
    p, g = rv.pcfg, rv.pcfg["gates"]
    d = {"channel": ch, "as_of": rv.as_of, "current_budget": float(budget), "proposed_budget": float(budget), "action": None,
         "reasons": [], "stages": [], "evidence": {}, "increase_candidate": None}
    ie, hold, no_inc, no_dec = [], [], [], []

    S = load_summary(rv.st, ch, rv.as_of)
    if S is None:
        d.update(action="insufficient_evidence", reasons=["no_weekly_summary_for_this_review: the review has no forecast summary for this channel"])
        _stage(d, "completeness", False, "no weekly summary stored for this channel at this review", "all->insufficient_evidence")
        return d

    # 0. evidence binding: stale or inconsistent forecast evidence is never reused
    bind = verify_forecast_evidence(rv.st, S, rv.fcfg)
    d["evidence"]["forecast_evidence_binding"] = bind
    if not bind["ok"]:
        _stage(d, "evidence_binding", False, f"{bind['code']}: " + json.dumps({k: v for k, v in bind.items() if k not in ('ok', 'code')}, sort_keys=True),
               "all->insufficient_evidence")
        d.update(action="insufficient_evidence", reasons=[f"{bind['code']}: forecast evidence was not produced under the current forecast/quality configuration "
                                                            "or is internally inconsistent; regenerate it under the current configuration"])
        return d
    _stage(d, "evidence_binding", True, f"summary and its {bind['contributing_forecasts']} contributing forecasts carry the current configuration hash and agree")

    # 1. completeness (not the same as "all checks passed")
    ready = bool(S["fully_decision_ready"])
    _stage(d, "completeness", ready, "every expected window date is accounted for" if ready else f"not fully decision-ready: {json.loads(S['not_ready_reasons_json'])}",
           "none" if ready else "all->insufficient_evidence")
    if not ready:
        ie.append("window_not_fully_decision_ready")

    # 2. quality checks, each with its own blocking consequence
    quality = {c["name"]: c for c in json.loads(S["quality_json"])}
    for name, (scope, outcome) in QUALITY_BLOCKS.items():
        c = quality[name]
        if c["evaluable"] and not c["passed"]:
            _stage(d, f"quality:{name}", False, c["reason"], f"{scope}->{outcome}" if scope == "all" else scope)
            if scope == "all":
                (ie if outcome == "insufficient_evidence" else hold).append(f"quality_check_failed:{name}")
            else:
                no_inc.append(f"quality_check_failed:{name}")
        elif not c["evaluable"]:
            blocked = name in NOT_EVALUABLE_BLOCKS_INCREASE
            _stage(d, f"quality:{name}", True, "not evaluable: " + c["reason"], "increase" if blocked else "none")
            if blocked:
                no_inc.append(f"quality_check_not_evaluable:{name}")
        else:
            _stage(d, f"quality:{name}", True, c["reason"])

    # 3. forecast coverage (population consistency)
    pops = populations(rv, ch, S)
    d["evidence"]["populations"] = pops
    n_inc, share = pops["included"]["n_cohorts"], pops["included_spend_share"]
    cov_ok = n_inc >= g["min_included_cohorts"] and share is not None and share >= g["min_included_spend_share"]
    _stage(d, "forecast_coverage", cov_ok,
           f"{n_inc} of {pops['n_expected_dates']} window cohorts forecast (need {g['min_included_cohorts']}); included cohorts hold "
           f"{'n/a' if share is None else f'{100 * share:.0f}%'} of the window's decision-ready spend (need {100 * g['min_included_spend_share']:.0f}%)",
           "none" if cov_ok else "all->insufficient_evidence")
    if not cov_ok:
        ie.append("forecast_covers_too_little_of_the_window")
    spike_excluded = [e for e in pops["excluded_withheld"]["cohorts"] if e["reason"] == "revenue_spike"]
    if spike_excluded:
        no_dec.append("excluded_cohorts_withheld_for_revenue_spike: included-only average ROAS is biased low")
    _stage(d, "excluded_population", not spike_excluded, f"{len(pops['excluded_withheld']['cohorts'])} withheld cohort(s) excluded; "
           f"{len(spike_excluded)} for revenue_spike", "decrease" if spike_excluded else "none")

    # 4. uncertainty
    point = None if pd.isna(S["point_forecast"]) else float(S["point_forecast"])
    spend_inc = float(S["spend_included"])
    unc = {"interval_status": S["interval_status"], "calibration_n": int(S["calib_n"]), "interval_reason": S["interval_reason"],
           "note": "80% nominal empirical interval from earlier matured forecasts; observed coverage on the dev seed is below nominal, so it is not a guaranteed bound"}
    unc_ok = False
    if S["interval_status"] == "available" and point and point > 0:
        lo, hi = float(S["interval_lo"]), float(S["interval_hi"])
        width = (hi - lo) / point
        unc.update(lo=lo, hi=hi, width_ratio=width)
        unc_ok = width <= g["max_interval_width_ratio"]
        _stage(d, "uncertainty", unc_ok, f"interval width/forecast = {width:.2f} (limit {g['max_interval_width_ratio']})" if unc_ok
               else f"interval too wide: width/forecast {width:.2f} above {g['max_interval_width_ratio']}", "none" if unc_ok else "all->insufficient_evidence")
        if not unc_ok:
            ie.append("interval_too_wide")
    else:
        _stage(d, "uncertainty", False, f"no usable interval: {S['interval_reason']}", "all->insufficient_evidence")
        ie.append("no_usable_interval")
    d["evidence"]["uncertainty"] = unc

    # 5. economics: average ROAS, payback target
    avg = {"population": "included cohorts only", "label": "forecast-based average D30 ROAS (model-estimated: uses forecast revenue)",
           "forecast_d30_net_revenue": point, "spend": spend_inc,
           "point": eco.average_roas(point, spend_inc) if point else None}
    if unc_ok:
        avg["lower"], avg["upper"] = unc["lo"] / spend_inc, unc["hi"] / spend_inc
    d["evidence"]["average_d30_roas"] = avg
    target = eco.payback_target(rv.mature, p["payback"], ch)
    d["evidence"]["payback_target"] = {**target, "note": "estimate derived from synthetic mature history; payback_multiple x (sum D30 / sum D180)"}
    t = target["target_d30_roas"]
    _stage(d, "payback_target", t is not None, f"target D30 ROAS {t:.3f} ({target['level']} level, {target['n_cohorts']} mature cohorts)" if t is not None
           else f"no target: {target['reason']} {target.get('channel_history')}", "none" if t is not None else "all->insufficient_evidence")
    if t is None:
        ie.append("no_payback_target")

    # decision
    if ie:
        d.update(action="insufficient_evidence", reasons=ie)
        return d
    if hold:
        d.update(action="hold", reasons=hold)
        return d
    if not no_dec and avg["upper"] < t and budget > p["constraints"]["min_spend_floor"] + 1e-9:
        new = max(p["constraints"]["min_spend_floor"], budget * (1 - p["steps"]["decrease_step_pct"]))
        d.update(action="decrease", proposed_budget=round(float(new), 6),
                 reasons=[f"upper bound of average D30 ROAS {avg['upper']:.3f} is below the target {t:.3f} (policy heuristic on forecast average ROAS; not a guarantee of economic safety)"])
        return d
    if avg["point"] < t:  # heuristic precondition: under diminishing returns the marginal return does not exceed the average
        why = [f"average D30 ROAS {avg['point']:.3f} is below the target {t:.3f}, so (diminishing-returns heuristic) there is no basis for an increase"]
        if no_dec:
            why.append("decrease blocked: " + "; ".join(no_dec))
        elif avg["upper"] >= t:
            why.append(f"but its upper bound {avg['upper']:.3f} is not below the target, so a decrease is not justified")
        d.update(action="hold", reasons=why)
        return d
    # average ROAS point >= target: is a bounded increase TEST justified?
    if no_inc:
        d.update(action="hold", reasons=["an increase would be considered but is blocked: " + "; ".join(no_inc)])
        return d
    obs = rv.ann[(rv.ann.channel == ch) & (rv.ann.d3_complete == 1) & rv.ann.spend.notna() & (rv.ann.installs_reported > 0)]
    resp = eco.fit_response(obs.rename(columns={"installs_reported": "installs"})[["install_date", "spend", "installs"]], p["response"])
    d["evidence"]["response_curve"] = {**resp, "label": "model-estimated spend -> installs response (historical association, not causal incrementality)"}
    if not resp["usable"]:
        d.update(action="insufficient_evidence", reasons=["response curve unusable: " + "; ".join(resp["unusable_reasons"])])
        return d
    lo_r, hi_r = resp["spend_range"]
    if budget < lo_r or budget > hi_r:
        d.update(action="insufficient_evidence", reasons=[f"current budget {budget:.2f} is outside the observed spend range [{lo_r:.2f}, {hi_r:.2f}]: the curve cannot be used"])
        return d
    c = p["constraints"]
    s1 = min(budget * (1 + p["steps"]["increase_step_pct"]), budget * (1 + c["per_channel_change_cap"]), hi_r, ceiling * c["max_channel_share"])
    if s1 - budget < p["steps"]["min_step_fraction"] * budget:
        if hi_r - budget < p["steps"]["min_step_fraction"] * budget:
            d.update(action="insufficient_evidence", reasons=[f"no room for a step inside the observed spend range (top {hi_r:.2f}, budget {budget:.2f}): extrapolation is not supported"])
        else:
            d.update(action="hold", reasons=[f"the channel-share limit leaves no room for a step (budget {budget:.2f}, limit {ceiling * c['max_channel_share']:.2f})"])
        return d
    est = estimate_step(resp, budget, s1, S, point, unc, p)
    d["evidence"]["marginal_d30_roas_step"] = est
    d["increase_candidate"] = {"s1": float(s1), "lower": est["lower_bound"]["lower"], "point": est["finite_step"]["marginal_roas"]}
    if est["lower_bound"]["lower"] >= t:
        d.update(action="increase_test", proposed_budget=round(float(s1), 6),
                 reasons=[f"lower bound of model-estimated finite-step marginal D30 ROAS {est['lower_bound']['lower']:.3f} is at or above the target {t:.3f}; "
                          "this is a bounded TEST that needs a pre-registered experiment plan, not an optimal-allocation claim"])
    else:
        d.update(action="hold", reasons=[f"lower bound of model-estimated finite-step marginal D30 ROAS {est['lower_bound']['lower']:.3f} is below the target {t:.3f} "
                                          f"(point estimate {est['finite_step']['marginal_roas']:.3f})"])
    return d


def estimate_step(resp, s0, s1, S, point, unc, p) -> dict:
    """Finite-step marginal estimate and lower bound for raising daily spend from s0 to s1 (model-estimated)."""
    value = point / float(S["installs_included"])  # forecast D30 net revenue per install of the INCLUDED cohorts (matching population)
    rel_sd = ((unc["hi"] - unc["lo"]) / (2 * Z80)) / point
    fs = eco.finite_step(resp, s0, s1, value, p["marginal"]["incremental_quality_haircut"])
    lb = eco.marginal_lower_bound(resp, s0, s1, value, rel_sd, p["marginal"])
    return {"label": "MODEL-ESTIMATED finite-step marginal D30 ROAS (not a derivative, not causal evidence)",
            "assumptions": ["historical spend->installs association read as a response curve", "step stays inside the observed spend range",
                            f"incremental players are worth {p['marginal']['incremental_quality_haircut']:.0%} of the included cohorts' average forecast value per install",
                            "no spillover to organic or other channels"],
            "value_per_install_forecast_included": value, "value_relative_sd_from_interval": rel_sd, "finite_step": fs, "lower_bound": lb,
            "average_roas_at_current_budget_included": (point / float(S["spend_included"]))}


def decide_all(dec, st, as_of: str, budgets: dict, ceiling: float, pcfg: dict = None, fcfg: dict = None, review: Review = None) -> dict:
    """Decide every channel, then fund increase tests from the reserve in order of lower-bound marginal ROAS."""
    rv = review or build_review(dec, st, as_of, pcfg, fcfg)
    p = rv.pcfg
    decisions = {ch: decide_channel(rv, ch, budgets[ch], ceiling, budgets) for ch in sorted(budgets)}
    reserve_left = cons.reserve(budgets, ceiling)
    cands = sorted((d for d in decisions.values() if d["action"] == "increase_test"), key=lambda d: -d["increase_candidate"]["lower"])
    for d in cands:
        ch, s0 = d["channel"], d["current_budget"]
        t = d["evidence"]["payback_target"]["target_d30_roas"]
        room = min(d["proposed_budget"] - s0, reserve_left)
        if room < p["steps"]["min_step_fraction"] * s0:
            d.update(action="hold", proposed_budget=s0, reasons=[f"not enough unallocated reserve ({reserve_left:.2f}) to fund a step"])
            continue
        s1 = s0 + room
        if abs(s1 - d["proposed_budget"]) > 1e-9:  # reserve-limited step: re-estimate at the smaller step
            S = load_summary(rv.st, ch, as_of)
            est = estimate_step(d["evidence"]["response_curve"], s0, s1, S, d["evidence"]["average_d30_roas"]["forecast_d30_net_revenue"], d["evidence"]["uncertainty"], p)
            d["evidence"]["marginal_d30_roas_step"] = est
            if est["lower_bound"]["lower"] < t:
                d.update(action="hold", proposed_budget=s0, reasons=["lower bound of the marginal ROAS at the reserve-limited step is below the target"])
                continue
            d["proposed_budget"] = round(float(s1), 6)
        reserve_left -= d["proposed_budget"] - s0
    proposed = {ch: d["proposed_budget"] for ch, d in decisions.items()}
    return {"as_of": as_of, "decisions": decisions, "proposed_budgets": proposed,
            # the hash of the configuration these decisions were made under; the ledger refuses a proposal whose hash differs from its own
            "policy_hash": cons.policy_hash(p, rv.fcfg),
            "constraint_violations": cons.check_constraints(budgets, proposed, ceiling, p),
            # the exact inverse (proposed -> current) must also pass, so every proposed change can be rolled back immediately
            "inverse_violations": cons.check_constraints(proposed, budgets, ceiling, p),
            "totals": {"before": float(sum(budgets.values())), "after": float(sum(proposed.values())), "ceiling": float(ceiling),
                       "reserve_before": cons.reserve(budgets, ceiling), "reserve_after": cons.reserve(proposed, ceiling)}}


def build_rules_proposal(result: dict, ledger_version: int, pol_hash: str = None) -> Proposal:
    """Turn policy decisions into a Proposal. Increases carry qualification `plan_required` (no plan exists before Phase 4).
    The proposal is stamped with the hash of the configuration that PRODUCED the decisions (`result["policy_hash"]`); pass `pol_hash`
    only to override it in tests. A ledger holding a different configuration refuses the proposal as stale."""
    pol_hash = pol_hash or result["policy_hash"]
    items = []
    for ch, d in sorted(result["decisions"].items()):
        items.append(ProposalItem(
            channel=ch, action=d["action"], current_budget=d["current_budget"], new_budget=d["proposed_budget"],
            qualification="plan_required" if d["action"] == "increase_test" else "none", experiment_plan_hash=None,
            reason="; ".join(d["reasons"]), evidence=jsonable({k: v for k, v in d["evidence"].items() if k in
                                                              ("average_d30_roas", "payback_target", "marginal_d30_roas_step", "uncertainty")})))
    return Proposal(proposal_id=make_proposal_id(result["as_of"], items, ledger_version, pol_hash), as_of=result["as_of"],
                    ledger_version=ledger_version, policy_hash=pol_hash, source="rules_only", items=items)
