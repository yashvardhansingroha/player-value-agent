"""SIMULATED experiment plan: content hash, structural validation (exact budget aggregation), adequacy recomputed from numbers, binding."""
import copy

import pytest

from src.experiments import design as D
from src.experiments import plan as P
from tests.experiment_helpers import CFG, manual_plan


def test_valid_plan_has_no_structural_errors_and_a_stable_hash():
    p = manual_plan()
    assert P.validate_plan_structure(p) == []
    assert P.plan_hash(p) == P.plan_hash(copy.deepcopy(p)) and len(P.plan_hash(p)) == 64


@pytest.mark.parametrize("mutate", [
    lambda p: p["units"][0].__setitem__("planned_incremental_spend", p["units"][0]["planned_incremental_spend"] + 0.000001),
    lambda p: p["assignment"].__setitem__("seed", 8),
    lambda p: p["estimator"].__setitem__("alpha", 0.05),
    lambda p: p["guardrails"].__setitem__("max_cpi_ratio", 2.0),
    lambda p: p["decision_rule"].__setitem__("target_d30_roas", 0.51),
    lambda p: p["windows"].__setitem__("treatment_days", 15),
    lambda p: p.__setitem__("policy_hash", "b" * 64),
    lambda p: p["power_assessment"]["per_duration"]["14"]["alternatives"][0].__setitem__("keep_rate", 0.5),
    lambda p: p["budget_step"].__setitem__("proposed_daily_budget", 99.5),
    lambda p: p.__setitem__("channel", "tiktok"),
])
def test_any_change_to_any_part_of_the_plan_changes_its_hash(mutate):
    p = manual_plan()
    h = P.plan_hash(p)
    q = copy.deepcopy(p)
    mutate(q)
    assert P.plan_hash(q) != h


def test_treatment_unit_budgets_must_add_up_to_exactly_the_channel_step():
    p = manual_plan(baseline=101.7, proposed=111.87)
    treat = [u for u in p["units"] if u["arm"] == "treatment"]
    assert sum(P.micro(u["planned_incremental_spend"]) for u in treat) == P.micro(10.17)  # exact in millionths
    assert sum(P.micro(u["baseline_daily_budget"]) for u in p["units"]) == P.micro(101.7)  # units partition the channel budget
    assert len(treat) == len(p["units"]) // 2 and all(u["planned_incremental_spend"] == 0 for u in p["units"] if u["arm"] == "control")
    bad = copy.deepcopy(p)
    bad["units"][[u["arm"] for u in bad["units"]].index("treatment")]["planned_incremental_spend"] += 0.000001
    assert any("add up to the exact channel-level step" in e for e in P.validate_plan_structure(bad))
    short = copy.deepcopy(p)
    short["units"][0]["baseline_daily_budget"] += 0.01
    assert any("do not add up to the channel baseline budget" in e for e in P.validate_plan_structure(short))


def test_structure_checks_cover_windows_arms_and_the_decision_rule():
    p = manual_plan()
    cases = {
        "treatment must start on the decision date": lambda q: q["windows"].__setitem__("treatment_start", "2025-09-09"),
        "readout_available_on is not last acquisition day": lambda q: q["windows"].__setitem__("readout_available_on", "2025-09-30"),
        "pre-period outcomes would not be available": lambda q: q["windows"].__setitem__("reporting_lag_days", 40),
        "decision-rule target differs": lambda q: q["decision_rule"].__setitem__("target_d30_roas", 0.9),
        "unsupported estimator": lambda q: q["estimator"].__setitem__("name", "naive_difference"),
        "not an increase": lambda q: q["budget_step"].__setitem__("proposed_daily_budget", 80.0),
    }
    for needle, mutate in cases.items():
        q = copy.deepcopy(p)
        mutate(q)
        assert any(needle in e for e in P.validate_plan_structure(q)), needle
    small = manual_plan(k=8)  # 4 units per arm < min_units_per_arm 8
    assert any("fewer than 8 units in an arm" in e for e in P.validate_plan_structure(small))
    assert P.validate_plan_structure({"channel": "x"})[0].startswith("missing fields")


def test_adequacy_is_recomputed_from_the_recorded_numbers_not_trusted():
    ok = manual_plan(adequate=True)
    assert ok["power_assessment"]["adequate"] is True
    claimed = copy.deepcopy(manual_plan(adequate=False))
    claimed["power_assessment"]["adequate"] = True  # a plan that merely CLAIMS adequacy
    assert any("recorded adequacy" in e for e in P.validate_plan_structure(claimed))
    low_power = copy.deepcopy(ok)
    low_power["power_assessment"]["per_duration"]["14"]["alternatives"][4]["keep_rate"] = 0.5  # power at 1.5x target
    assert P.adequacy_from_numbers(low_power["power_assessment"])[0] is False
    leaky = copy.deepcopy(ok)
    leaky["power_assessment"]["per_duration"]["14"]["alternatives"][2]["keep_rate"] = 0.30  # false-keep at the target boundary
    ok2, reasons = P.adequacy_from_numbers(leaky["power_assessment"])
    assert ok2 is False and "false-keep at 1.0x target" in reasons[0]
    tolerated = copy.deepcopy(ok)
    tolerated["power_assessment"]["per_duration"]["14"]["alternatives"][2].update(keep_rate=0.115, mc_se=0.01)  # 0.10 + 2 MC SE = 0.12
    assert P.adequacy_from_numbers(tolerated["power_assessment"])[0] is True
    assert P.adequacy_from_numbers(manual_plan(adequate=False)["power_assessment"])[0] is False


def test_a_plan_that_follows_the_configuration_passes_and_one_under_another_configuration_does_not():
    p = manual_plan()
    assert P.validate_plan_against_config(p, CFG) == []
    assert P.validate_plan_against_config(manual_plan(adequate=False), CFG) == []  # an inadequate plan is still a faithful record
    assert P.validate_plan_against_config(manual_plan(treatment_days=28), CFG) == []  # a longer chosen window: shorter ones record as inadequate
    lowered = copy.deepcopy(CFG)
    lowered["power"]["target_power"] = 0.01
    assert P.validate_plan_against_config(p, lowered) == ["the authoritative experiment configuration does not hash to the plan's experiment_config_hash"]
    assert P.config_hash(CFG) == D.config_hash(CFG) == p["experiment_config_hash"] and P.config_hash(lowered) != P.config_hash(CFG)


@pytest.mark.parametrize("needle,mutate", [
    ("adequacy standards differ", lambda p: p["power_assessment"]["standards"].__setitem__("target_power", 0.01)),
    ("adequacy standards differ", lambda p: p["power_assessment"]["standards"].__setitem__("below_boundary_multiples", [])),
    ("adequacy standards differ", lambda p: p["power_assessment"]["standards"].__setitem__("false_keep_tolerance_mc_se", 50.0)),
    ("estimator settings differ", lambda p: p["estimator"].__setitem__("alpha", 0.25)),
    ("estimator settings differ", lambda p: p["estimator"].__setitem__("max_delivery_fraction", 99.0)),
    ("estimator settings differ", lambda p: p["estimator"].__setitem__("critical_value", "normal")),
    ("guardrail settings differ", lambda p: p["guardrails"].__setitem__("max_cpi_ratio", 3.0)),
    ("guardrail settings differ", lambda p: p["guardrails"].__setitem__("min_d1_retention_change", -0.5)),
    ("decision-rule alpha differs", lambda p: p["decision_rule"].__setitem__("alpha", 0.25)),
    ("is not an allowed duration", lambda p: p.__setitem__("windows", D.windows(p["decision_as_of"], 2, 14, 21))),
    ("pre-period length", lambda p: p.__setitem__("windows", D.windows(p["decision_as_of"], 2, 7, 14))),
    ("not the supported D30", lambda p: p["windows"].update(outcome_horizon_days=7, readout_available_on=P.add_days(p["windows"]["treatment_end"], 9))),
    ("replicates/seed differ", lambda p: p["power_assessment"].__setitem__("seed", 1)),
    ("not for all allowed durations", lambda p: p["power_assessment"]["per_duration"].pop("56")),
    ("wrong duration or replicate count", lambda p: p["power_assessment"]["per_duration"]["28"].__setitem__("replicates", 300)),
    ("all the configured multiples", lambda p: p["power_assessment"]["per_duration"]["28"]["alternatives"].pop()),
    ("is not a count out of", lambda p: p["power_assessment"]["per_duration"]["28"]["alternatives"][4].__setitem__("keep_rate", 0.9004)),
    ("is not binomial", lambda p: p["power_assessment"]["per_duration"]["28"]["alternatives"][2].__setitem__("mc_se", 0.2)),
    ("not multiple x this plan's target", lambda p: p["power_assessment"]["per_duration"]["28"]["alternatives"][4].__setitem__("true_iroas", 99.0)),
    ("is not the shortest window", lambda p: p["power_assessment"].__setitem__("chosen_treatment_days", 28)),
    ("THIS design", lambda p: p["power_assessment"].__setitem__("design_fingerprint", "0" * 64)),
])
def test_every_part_of_a_plan_must_follow_the_authoritative_configuration(needle, mutate):
    p = manual_plan(treatment_days=14)
    assert p["experiment_config_hash"] == D.config_hash(CFG)  # the declared configuration hash stays the ORIGINAL one in every case
    mutate(p)
    errors = P.validate_plan_against_config(p, CFG)
    assert any(needle in e for e in errors), (needle, errors)


def test_numbers_recorded_for_another_design_are_detected_by_the_fingerprint():
    p = manual_plan()
    for change in (lambda q: q["budget_step"].update(proposed_daily_budget=99.5, step_amount=9.5),  # another step
                   lambda q: q["assignment"]["treated"].reverse() or q["assignment"].update(seed=99),  # another assignment
                   lambda q: q["units"][0].update(baseline_daily_budget=q["units"][0]["baseline_daily_budget"] + 1e-6),  # another unit budget
                   lambda q: q["target"].update(target_d30_roas=0.4)):  # another target
        q = copy.deepcopy(p)
        change(q)
        assert P.design_fingerprint(q) != P.design_fingerprint(p)
    assert P.design_fingerprint(copy.deepcopy(p)) == P.design_fingerprint(p)  # deterministic, and independent of the recorded numbers themselves
    q = copy.deepcopy(p)
    q["power_assessment"]["per_duration"]["14"]["alternatives"][0]["keep_rate"] = 0.5
    assert P.design_fingerprint(q) == P.design_fingerprint(p)


def test_an_informational_extra_multiple_is_allowed_but_is_held_to_the_same_consistency_checks():
    p = manual_plan()
    extra = dict(p["power_assessment"]["per_duration"]["14"]["alternatives"][2], multiple=1.2, true_iroas=1.2 * 0.5)  # like the real study's marginal-ROAS row
    for blk in p["power_assessment"]["per_duration"].values():
        blk["alternatives"].append(dict(extra))
    assert P.validate_plan_against_config(p, CFG) == []
    p["power_assessment"]["per_duration"]["14"]["alternatives"][-1]["mc_se"] = 0.3
    assert any("is not binomial" in e for e in P.validate_plan_against_config(p, CFG))


def test_malformed_plan_content_is_reported_not_raised():
    p = manual_plan()
    del p["power_assessment"]["per_duration"]["14"]["alternatives"][0]["keep_rate"]
    errors = P.validate_plan_against_config(p, CFG)
    assert errors and "malformed" in errors[0]


def item(plan, **over):
    it = {"channel": plan["channel"], "current_budget": plan["budget_step"]["baseline_daily_budget"],
          "new_budget": plan["budget_step"]["proposed_daily_budget"], "evidence": {"payback_target": {"target_d30_roas": plan["target"]["target_d30_roas"]}}}
    it.update(over)
    return it


def test_binding_requires_matching_channel_step_state_policy_date_and_target():
    p = manual_plan(policy_hash="c" * 64)
    state = {"budgets": {"meta": 90.0}}
    assert P.validate_plan_binding(p, item(p), state, "c" * 64, "2025-09-08") == []
    assert any("channel" in e for e in P.validate_plan_binding(p, item(p, channel="tiktok"), {"budgets": {"tiktok": 90.0}}, "c" * 64, "2025-09-08"))
    assert any("proposed budget" in e for e in P.validate_plan_binding(p, item(p, new_budget=108.0), state, "c" * 64, "2025-09-08"))
    assert any("baseline budget" in e for e in P.validate_plan_binding(p, item(p), {"budgets": {"meta": 91.0}}, "c" * 64, "2025-09-08"))
    assert any("different policy configuration" in e for e in P.validate_plan_binding(p, item(p), state, "d" * 64, "2025-09-08"))
    assert any("decision date" in e for e in P.validate_plan_binding(p, item(p), state, "c" * 64, "2025-09-15"))
    assert any("target" in e for e in P.validate_plan_binding(p, item(p, evidence={"payback_target": {"target_d30_roas": 0.6}}), state, "c" * 64, "2025-09-08"))
