"""Policy gates and decisions on a synthetic review where each factor can be changed alone.

Baseline situation (channel 'meta', budget 80/day): 7 window cohorts, each 80 installs and 80 spend, forecast value 2.0 per install,
so average D30 ROAS = 2.0; interval +/-20%; payback history D180/D30 = 2 -> target D30 ROAS 0.5; response installs = 80 x (spend/80)^0.7,
so the model-estimated finite-step marginal ROAS for 80 -> 88 is about 1.1 (point) and about 0.9 (lower bound): an increase test."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from src.forecast import store
from src.policy import economics as eco
from src.policy import policy as pol
from src.policy.constraints import check_constraints, load_forecast_config, load_policy

AS_OF = "2025-09-08"
WINDOW = pd.date_range("2025-08-28", "2025-09-03").strftime("%Y-%m-%d").tolist()
QUALITY_NAMES = ["window_min_installs", "missing_days", "reporting_completeness", "revenue_concentration", "cpi_anomaly", "tracking_gap"]


def history(channel, constant=False, n=140):
    dates = pd.date_range(end="2025-09-03", periods=n)
    spend = np.full(n, 80.0) if constant else 80 * np.exp(0.3 * np.sin(np.arange(n) / 4.0) + 0.1 * np.cos(np.arange(n) / 1.7))
    installs = 80 * (spend / 80) ** 0.7
    df = pd.DataFrame({"channel": channel, "install_date": dates.strftime("%Y-%m-%d"), "d3_complete": 1, "spend": spend,
                       "installs_reported": installs.round().astype(int), "d0_3_net_revenue": 40.0})
    in_window = df.install_date.isin(WINDOW)  # the window cohorts match the summary: 80 spend and 80 installs each
    df.loc[in_window, "spend"], df.loc[in_window, "installs_reported"] = 80.0, 80
    return df


def insert_contributor(st, ch, date, point, config_hash):
    row = {"channel": ch, "install_date": date, "issued_as_of": AS_OF, "method": "channel_multiplier", "features_json": "{}", "feature_hash": "f",
           "training_json": "{}", "train_n": 30, "train_max_outcome_complete_at": "2025-08-01", "point_forecast": point, "reference_forecast": point,
           "observed_d3_floor": 0.0, "point_floor_adjusted": 0, "interval_status": "unavailable", "interval_lo": None, "interval_hi": None,
           "interval_floor_adjusted": 0, "interval_reason": "x", "calib_n": 0, "calib_max_outcome_complete_at": None, "config_hash": config_hash}
    st.execute(f"INSERT INTO forecast_issued ({','.join(row)}) VALUES ({','.join('?' * len(row))})", tuple(row.values()))


def build(channels=("meta",), value=2.0, n_included=7, withheld_reason="below_min_installs", ready=True, quality=None, interval="ok",
          mature_n=40, y180=200.0, constant_spend=False, pcfg=None, dates=None, summary_hash=None, contributor_hash=None,
          drop_contributor=False, point_skew=0.0, fcfg=None):
    """Return the review. `quality` maps check name -> 'fail' | 'not_evaluable'. The weekly summary carries the evidence hash of `fcfg`
    (override with `summary_hash`) and its contributing forecasts are stored too (`contributor_hash`, `drop_contributor`, `point_skew`
    make them inconsistent on purpose)."""
    fcfg = fcfg or load_forecast_config()
    ehash = store.forecast_evidence_hash(fcfg)
    st = store.create_store(":memory:")
    ann, mature = [], []
    for ch in channels:
        ann.append(history(ch, constant_spend))
        mature.append(pd.DataFrame({"channel": ch, "install_date": [f"2025-01-{i % 28 + 1:02d}-{i}" for i in range(mature_n)], "installs": 100, "y30": 100.0, "y180": y180}))
        inc, held = WINDOW[:n_included], WINDOW[n_included:]
        acc = [{"install_date": d, "status": "forecast"} for d in inc] + [{"install_date": d, "status": "withheld", "reason": withheld_reason} for d in held]
        spend_inc, installs_inc = 80.0 * len(inc), 80 * len(inc)
        point = value * installs_inc
        checks = []
        for name in QUALITY_NAMES:
            state = (quality or {}).get(name)
            checks.append({"name": name, "passed": state is None, "evaluable": state != "not_evaluable", "reason": f"{name}: {state or 'ok'}", "evidence": {}})
        iv = {"ok": (point * 0.8, point * 1.2, "available", None), "wide": (point * 0.1, point * 2.0, "available", None),
              "none": (None, None, "unavailable", "insufficient_calibration_history: 3 matured earlier forecasts, need 12")}[interval]
        row = {"channel": ch, "as_of": AS_OF, "window_start": WINDOW[0], "window_end": WINDOW[-1], "included_json": json.dumps(inc),
               "withheld_json": json.dumps([]), "accounting_json": json.dumps(acc), "fully_decision_ready": int(ready),
               "not_ready_reasons_json": json.dumps([] if ready else ["1 install date(s) inside the window not decision-ready"]),
               "n_expected_dates": 7, "n_included": len(inc), "n_withheld": len(held), "installs_included": installs_inc, "spend_included": spend_inc,
               "point_forecast": point, "observed_d3_floor": 0.0, "interval_status": iv[2], "interval_lo": iv[0], "interval_hi": iv[1],
               "interval_floor_adjusted": 0, "interval_reason": iv[3], "calib_n": 30, "quality_json": json.dumps(checks), "config_hash": summary_hash or ehash}
        st.execute(f"INSERT INTO weekly_summary ({','.join(row)}) VALUES ({','.join('?' * len(row))})", tuple(row.values()))
        for k, date in enumerate(inc):  # one stored forecast per included cohort: 80 installs x value each
            if drop_contributor and k == 0:
                continue
            insert_contributor(st, ch, date, value * 80 + (point_skew if k == 0 else 0.0), contributor_hash or ehash)
    rv = pol.Review(AS_OF, None, st, pcfg or load_policy(), fcfg, None, pd.concat(ann, ignore_index=True), pd.concat(mature, ignore_index=True))
    return rv


def decide(rv, budget=80.0, ceiling=480.0, ch="meta"):
    return pol.decide_channel(rv, ch, budget, ceiling, {ch: budget})


def stage(d, name):
    return next(s for s in d["stages"] if s["stage"] == name)


def test_baseline_is_a_bounded_increase_test_with_labelled_model_estimated_evidence():
    d = decide(build())
    assert d["action"] == "increase_test" and d["proposed_budget"] == pytest.approx(88.0)  # +10%, inside cap, range, share
    est = d["evidence"]["marginal_d30_roas_step"]
    assert "MODEL-ESTIMATED" in est["label"] and "not a derivative" in est["label"]
    fs = est["finite_step"]
    assert fs["marginal_roas"] == pytest.approx(1.1, abs=0.05) and fs["derivative_marginal_roas"] != pytest.approx(fs["marginal_roas"])
    assert est["lower_bound"]["lower"] >= d["evidence"]["payback_target"]["target_d30_roas"] == pytest.approx(0.5)
    assert est["lower_bound"]["lower"] < fs["marginal_roas"]  # the bound is below the point estimate
    assert any("incremental players" in a for a in est["assumptions"])
    assert "not a guaranteed" in d["evidence"]["uncertainty"]["note"]  # intervals are not described as guaranteed coverage
    assert "pre-registered experiment plan" in d["reasons"][0] and "optimal" in d["reasons"][0]


def test_population_consistency_average_roas_uses_included_cohorts_and_excluded_are_reported_without_their_d30():
    rv = build(n_included=5, withheld_reason="revenue_spike")
    d = decide(rv)
    pops, avg = d["evidence"]["populations"], d["evidence"]["average_d30_roas"]
    assert pops["included"]["n_cohorts"] == 5 and pops["included"]["spend"] == pytest.approx(400.0) and "ONLY the included cohorts" in pops["note"]
    assert avg["spend"] == pytest.approx(400.0) and avg["population"] == "included cohorts only"
    assert avg["point"] == pytest.approx(2.0 * 400 / 400)  # forecast revenue and spend of the SAME cohorts
    exc = pops["excluded_withheld"]
    assert exc["n_cohorts"] == 2 and exc["spend"] == pytest.approx(160.0) and pops["included_spend_share"] == pytest.approx(400 / 560)
    for c in exc["cohorts"]:
        assert set(c) == {"install_date", "reason", "installs", "spend", "observed_d0_3_net_revenue"}  # decision-date facts only
        assert not any("d30" in k for k in c)


def test_matching_evidence_is_bound_and_used():
    d = decide(build())
    assert d["action"] == "increase_test" and stage(d, "evidence_binding")["passed"] is True
    b = d["evidence"]["forecast_evidence_binding"]
    assert b["ok"] and b["contributing_forecasts"] == 7 and b["expected_hash"] == store.forecast_evidence_hash(load_forecast_config())


def test_summary_made_under_a_different_forecast_configuration_is_refused_with_both_hashes():
    """Reproducer from independent review: the stored summary was written under summary_min_calibration = 12; the review now runs
    under 1000. The old behaviour still recommended an increase; the evidence must be regenerated instead."""
    fc = copy.deepcopy(load_forecast_config())
    stored = store.forecast_evidence_hash(fc)
    fc["forecast"]["summary_min_calibration"] = 1000
    rv = build(summary_hash=stored)  # evidence produced under the ORIGINAL configuration
    rv.fcfg = fc  # ... but this review runs under the changed one
    d = decide(rv)
    assert d["action"] == "insufficient_evidence" and d["proposed_budget"] == 80.0
    assert d["reasons"][0].startswith("forecast_evidence_config_mismatch") and "regenerate" in d["reasons"][0]
    b = d["evidence"]["forecast_evidence_binding"]
    assert b["ok"] is False and b["stored_hash"] == stored and b["expected_hash"] == store.forecast_evidence_hash(fc) and b["stored_hash"] != b["expected_hash"]
    assert [s["stage"] for s in d["stages"]] == ["evidence_binding"] and d["stages"][0]["passed"] is False  # nothing else was evaluated on stale evidence
    assert "marginal_d30_roas_step" not in d["evidence"] and "average_d30_roas" not in d["evidence"]
    assert decide(rv_with(build(value=0.3, summary_hash=stored), fc))["action"] == "insufficient_evidence"  # a clearly unprofitable channel is not cut on stale evidence either


def rv_with(rv, fc):
    rv.fcfg = fc
    return rv


def test_a_forecast_setting_that_does_not_affect_evidence_does_not_invalidate_it():
    fc = copy.deepcopy(load_forecast_config())
    fc["backtest"]["last_as_of_day"] = 999  # run scheduling is recorded separately (store_meta); it is outside the evidence hash by design
    assert decide(rv_with(build(), fc))["action"] == "increase_test"


def test_summary_with_foreign_hash_is_refused():
    d = decide(build(summary_hash="foreign"))
    assert d["action"] == "insufficient_evidence" and d["evidence"]["forecast_evidence_binding"]["stored_hash"] == "foreign"


def test_contributing_forecasts_must_agree_with_the_summary():
    other = decide(build(contributor_hash="other-config"))  # summary hash matches but its contributors were made under another configuration
    b = other["evidence"]["forecast_evidence_binding"]
    assert other["action"] == "insufficient_evidence" and b["code"] == "forecast_evidence_inconsistent" and len(b["contributing_forecasts_with_other_config"]) == 7
    missing = decide(build(drop_contributor=True))["evidence"]["forecast_evidence_binding"]
    assert missing["code"] == "forecast_evidence_inconsistent" and len(missing["contributing_forecasts_missing"]) == 1
    skew = decide(build(point_skew=25.0))["evidence"]["forecast_evidence_binding"]
    assert skew["code"] == "forecast_evidence_inconsistent" and skew["sum_of_contributing_forecasts"] == pytest.approx(skew["summary_point_forecast"] + 25.0)


def test_completeness_is_separate_from_quality_and_blocks_everything():
    d = decide(build(ready=False))
    assert d["action"] == "insufficient_evidence" and "window_not_fully_decision_ready" in d["reasons"]
    assert stage(d, "completeness")["passed"] is False and stage(d, "completeness")["blocks"] == "all->insufficient_evidence"
    ok = decide(build())
    assert stage(ok, "completeness")["passed"] is True  # fully_decision_ready does not mean "all checks passed"
    with_failed_check = decide(build(quality={"revenue_concentration": "fail"}))
    assert stage(with_failed_check, "completeness")["passed"] is True and stage(with_failed_check, "quality:revenue_concentration")["passed"] is False


@pytest.mark.parametrize("check,expected", [("tracking_gap", "insufficient_evidence"), ("window_min_installs", "insufficient_evidence"),
                                            ("missing_days", "insufficient_evidence"), ("reporting_completeness", "insufficient_evidence"),
                                            ("cpi_anomaly", "hold")])
def test_failures_that_block_every_change(check, expected):
    d = decide(build(quality={check: "fail"}))
    assert d["action"] == expected and f"quality_check_failed:{check}" in d["reasons"]
    low = decide(build(quality={check: "fail"}, value=0.3))  # even a clearly unprofitable channel is not cut on bad data
    assert low["action"] == expected


def test_concentration_blocks_increases_only_and_not_evaluable_checks_block_increases_only():
    assert decide(build(quality={"revenue_concentration": "fail"}))["action"] == "hold"  # increase blocked
    assert decide(build(quality={"revenue_concentration": "fail"}, value=0.3))["action"] == "decrease"  # decrease still allowed
    d = decide(build(quality={"tracking_gap": "not_evaluable"}))
    assert d["action"] == "hold" and "quality_check_not_evaluable:tracking_gap" in d["reasons"][0]
    assert decide(build(quality={"cpi_anomaly": "not_evaluable"}, value=0.3))["action"] == "decrease"


def test_forecast_coverage_gate_counts_cohorts_and_spend_share():
    d = decide(build(n_included=3))
    assert d["action"] == "insufficient_evidence" and "forecast_covers_too_little_of_the_window" in d["reasons"]
    assert decide(build(n_included=4))["action"] == "insufficient_evidence"  # 4 cohorts but only 57% of the spend
    assert decide(build(n_included=5))["action"] == "increase_test"  # 5/7 = 71% >= 70%


def test_spike_excluded_cohorts_block_decreases_because_included_only_roas_is_biased_low():
    spike = decide(build(n_included=5, withheld_reason="revenue_spike", value=0.3))
    assert spike["action"] == "hold" and "decrease blocked" in " ".join(spike["reasons"]) and "revenue_spike" in " ".join(spike["reasons"])
    assert stage(spike, "excluded_population")["blocks"] == "decrease"
    other = decide(build(n_included=5, withheld_reason="below_min_installs", value=0.3))
    assert other["action"] == "decrease" and other["proposed_budget"] == pytest.approx(72.0)  # -10%
    assert decide(build(n_included=5, withheld_reason="revenue_spike"))["action"] == "increase_test"  # increases are not blocked by it


def test_uncertainty_must_exist_and_be_usable():
    none = decide(build(interval="none"))
    assert none["action"] == "insufficient_evidence" and "no_usable_interval" in none["reasons"]
    wide = decide(build(interval="wide"))
    assert wide["action"] == "insufficient_evidence" and "interval_too_wide" in wide["reasons"]


def test_missing_history_gives_insufficient_evidence_with_the_reason():
    d = decide(build(mature_n=10))
    assert d["action"] == "insufficient_evidence" and d["reasons"] == ["no_payback_target"]
    assert d["evidence"]["payback_target"]["reason"] == "insufficient_history"


def test_unusable_response_curve_blocks_increases_but_not_decreases():
    d = decide(build(constant_spend=True))
    assert d["action"] == "insufficient_evidence" and "response curve unusable" in d["reasons"][0] and "barely varied" in d["reasons"][0]
    assert decide(build(constant_spend=True, value=0.3))["action"] == "decrease"  # a cut does not need the curve


def test_observed_spend_range_is_enforced():
    rv = build()
    obs = rv.ann[rv.ann.channel == "meta"]
    hi = eco.fit_response(obs.rename(columns={"installs_reported": "installs"}), rv.pcfg["response"])["spend_range"][1]
    outside = decide(rv, budget=hi * 1.5)
    assert outside["action"] == "insufficient_evidence" and "outside the observed spend range" in outside["reasons"][0]
    clipped = decide(rv, budget=hi / 1.05)  # a 10% step would leave the range: the step is shortened to the top of the range
    assert clipped["action"] == "increase_test" and clipped["proposed_budget"] == pytest.approx(hi)
    no_room = decide(rv, budget=hi * 0.995)
    assert no_room["action"] == "insufficient_evidence" and "extrapolation is not supported" in no_room["reasons"][0]


def test_marginal_lower_bound_below_target_holds_and_so_does_a_low_average_roas():
    weak = copy.deepcopy(load_policy())
    weak["marginal"]["incremental_quality_haircut"] = 0.2  # incremental players assumed nearly worthless
    d = decide(build(pcfg=weak))
    assert d["action"] == "hold" and "below the target" in d["reasons"][0] and "lower bound" in d["reasons"][0]
    mid = decide(build(value=0.45))  # average 0.45 < target 0.5, upper bound 0.54 >= target: neither cut nor increase
    assert mid["action"] == "hold" and "not below the target" in " ".join(mid["reasons"])
    assert decide(build(value=0.3))["action"] == "decrease"  # upper bound 0.36 < 0.5


def test_decrease_never_goes_below_the_floor_and_respects_the_step():
    assert decide(build(value=0.3), budget=10.0)["action"] == "hold"  # already at the floor
    d = decide(build(value=0.3), budget=10.5)
    assert d["action"] == "decrease" and d["proposed_budget"] == pytest.approx(10.0)  # 10% would be 9.45: held at the floor


def test_no_summary_means_insufficient_evidence():
    rv = build()
    d = pol.decide_channel(rv, "unity", 80.0, 480.0, {"unity": 80.0})
    assert d["action"] == "insufficient_evidence" and "no_weekly_summary" in d["reasons"][0]


def multi(ceiling):
    """meta and unity are both strong (identical numbers, so funding order falls back to name); tiktok has no summary."""
    rv = build(channels=("meta", "unity"))
    budgets = {"meta": 80.0, "tiktok": 80.0, "unity": 80.0}
    return pol.decide_all(None, rv.st, AS_OF, budgets, ceiling, review=rv), budgets


def test_increase_tests_are_funded_from_the_reserve_only_and_totals_stay_under_the_ceiling():
    res, budgets = multi(ceiling=250.0)  # total 240, reserve 10; two strong channels each want +8
    d = res["decisions"]
    funded = sorted((c, x["proposed_budget"]) for c, x in d.items() if x["action"] == "increase_test")
    assert len(funded) == 2 and sum(b - 80.0 for _, b in funded) == pytest.approx(10.0)  # 8 + 2: the reserve is exactly used
    assert res["totals"]["reserve_before"] == pytest.approx(10.0) and res["totals"]["reserve_after"] == pytest.approx(0.0)
    assert res["totals"]["after"] <= res["totals"]["ceiling"] + 1e-9 and res["constraint_violations"] == []
    assert d["tiktok"]["action"] == "insufficient_evidence" and d["tiktok"]["proposed_budget"] == 80.0  # no summary: untouched
    tight, _ = multi(ceiling=245.0)  # reserve 5: only the first candidate gets a (shortened) step
    acts = {c: x["action"] for c, x in tight["decisions"].items()}
    assert sorted(a for a in acts.values()) == ["hold", "increase_test", "insufficient_evidence"]
    held = next(x for x in tight["decisions"].values() if x["action"] == "hold")
    assert "not enough unallocated reserve" in held["reasons"][0]


def test_rules_proposal_marks_increases_as_needing_a_plan_and_is_hash_stable():
    res, budgets = multi(ceiling=250.0)
    p1 = pol.build_rules_proposal(res, 1, "abc")
    p2 = pol.build_rules_proposal(res, 1, "abc")
    assert p1.proposal_id == p2.proposal_id
    inc = [i for i in p1.items if i.action == "increase_test"]
    assert inc and all(i.qualification == "plan_required" and i.experiment_plan_hash is None for i in inc)
    assert all(i.qualification == "none" for i in p1.items if i.action != "increase_test")
    assert check_constraints(budgets, {i.channel: i.new_budget for i in p1.items}, 250.0, load_policy()) == []
