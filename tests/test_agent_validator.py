"""Phase 5 validator: fabricated citations, wrong values, wrong channel/population/call, stale evidence, forbidden actions, missing and inadequate plans,
changed steps, hard constraints. Synthetic hand-made review; MockLLM-free (the validator is called directly on scripted answers)."""
import copy

import pytest

from src.agent import tools as T
from src.agent import validator as V
from src.agent.llm import ToolCall
from src.experiments import plan as P
from tests.agent_helpers import AS_OF, Script, base_items, claim, make_ctx, output, standard_conversation


def validate(ctx, mutate=None, calls_extra=(), summary=None, tamper=None):
    s = Script(ctx, extra_calls=[("plan_experiment", {"channel": "applovin", "new_budget": 110.0}), *calls_extra])
    runner = T.ToolRunner(ctx, 50)
    for name, args in s.calls:
        runner.execute(ToolCall(name, args))
    if tamper:
        tamper(runner)
    items = base_items(s)
    if mutate:
        items = mutate(items, s) or items
    kw = {} if summary is None else {"summary": summary}
    out, errs = V.parse_output(output(items, **kw))
    assert out is not None, errs
    return V.validate_agent_output(out, ctx, runner), s, runner


def codes(rep):
    return sorted({e["code"] for e in rep["errors"]})


def item(items, ch):
    return next(i for i in items if i["channel"] == ch)


def test_a_fully_cited_proposal_that_follows_the_policy_is_valid_and_lists_the_plan_to_register():
    ctx = make_ctx()
    rep, s, _ = validate(ctx)
    assert rep["ok"] and rep["errors"] == []
    assert rep["plans_to_register"] == [ctx.qualification("applovin", 110.0)["plan_hash"]]  # registration and approval stay outside the agent


# ------------------------------------------------------------------------------------------------ citations


def test_a_fabricated_tool_call_id_is_rejected():
    rep, *_ = validate(make_ctx(), lambda items, s: item(items, "meta")["claims"].append(claim("tc_099", "action", "hold", "policy_decision", "meta")))
    assert "unknown_tool_call" in codes(rep) and not rep["ok"]


def test_a_wrong_value_is_rejected_even_when_the_field_exists():
    def wrong(items, s):
        item(items, "applovin")["claims"][2]["value"] = 0.9  # recalculation.lower_bound is 0.62
    rep, *_ = validate(make_ctx(), wrong)
    assert "value_mismatch" in codes(rep)


def test_correct_rounding_is_accepted_but_a_coarse_or_wrong_rounding_is_not():
    def rounded(items, s):
        item(items, "applovin")["claims"][2]["value"] = 0.6  # 0.62 written with one decimal is a correct rounding within 0.005? it is 0.02 away
    rep, *_ = validate(make_ctx(), rounded)
    assert "value_mismatch" in codes(rep)  # 0.6 is not within 0.005 / 2% of 0.62
    assert V.number_matches(0.62, 0.6234, 2) and V.number_matches(0.53, 0.527, 2) and not V.number_matches(0.0, 0.4, 0) and not V.number_matches(0.63, 0.6234, 2)
    assert V.value_matches(True, True) and not V.value_matches(1.0, True) and not V.value_matches("hold", "decrease")
    # signs: a value near zero is within the rounding tolerance of its negative, so the sign must be compared explicitly
    assert not V.number_matches(-0.001, 0.001, 3) and not V.number_matches(0.001, -0.001, 3) and V.number_matches(-0.001, -0.001, 3) and V.number_matches(0.0, 0.001, 0)
    assert not V.value_matches(-0.002, 0.002) and V.value_matches(-0.002, -0.002)


def test_evidence_from_another_channel_is_rejected():
    def other_channel(items, s):
        # tiktok's item cites meta's policy decision call: the numbers/actions are real but they belong to another channel
        item(items, "tiktok")["claims"] = [claim(s.id("get_policy_decision", "meta"), "action", "hold", "policy_decision", "meta")]
    rep, *_ = validate(make_ctx(), other_channel)
    assert "claim_wrong_channel" in codes(rep) and "missing_required_evidence" in codes(rep)
    def lying_about_the_channel(items, s):
        item(items, "tiktok")["claims"] = [claim(s.id("get_policy_decision", "meta"), "action", "hold", "policy_decision", "tiktok")]
    rep2, *_ = validate(make_ctx(), lying_about_the_channel)
    assert "claim_wrong_channel" in codes(rep2)


def test_a_matching_number_from_the_wrong_population_or_cohort_is_not_evidence():
    ctx = make_ctx()
    calls = [("get_forecast_and_populations", {"channel": "meta"})]

    def mislabelled(items, s):
        tc = "tc_%03d" % len(s.calls)  # the forecast call is the last extra call
        item(items, "meta")["claims"].append(claim(tc, "excluded_withheld.installs", 313, "included_cohorts", "meta"))  # right number, wrong population label
    rep, *_ = validate(ctx, mislabelled, calls_extra=calls)
    assert "wrong_population" in codes(rep)

    def wrong_cohort(items, s):
        tc = "tc_%03d" % len(s.calls)  # the forecast call is the last extra call
        item(items, "meta")["claims"].append(claim(tc, "included.installs", 313, "included_cohorts", "meta"))  # the EXCLUDED cohorts' number presented as the included one
    rep2, *_ = validate(ctx, wrong_cohort, calls_extra=calls)
    assert "value_mismatch" in codes(rep2)

    def right(items, s):
        tc = "tc_%03d" % len(s.calls)  # the forecast call is the last extra call
        item(items, "meta")["claims"] += [claim(tc, "included.installs", 700, "included_cohorts", "meta"), claim(tc, "excluded_withheld.installs", 313, "excluded_withheld_cohorts", "meta")]
        item(items, "meta")["reason"] = "Policy action hold. The included cohorts have 700 installs and the excluded withheld cohorts 313."
    rep3, *_ = validate(ctx, right, calls_extra=calls)
    assert rep3["ok"], rep3["errors"]


def test_a_number_from_the_wrong_tool_call_is_not_evidence():
    def wrong_call(items, s):  # cite tiktok's recalculation for applovin
        item(items, "applovin")["claims"][2]["tool_call_id"] = s.id("get_policy_decision", "applovin")
    rep, *_ = validate(make_ctx(), wrong_call)
    assert codes(rep)  # that call has no such field / wrong population
    assert "field_not_found" in codes(rep)


def test_stale_evidence_is_rejected_by_date_and_by_configuration():
    def old_date(runner):
        runner.records["tc_003"]["as_of"] = "2025-10-06"
    rep, *_ = validate(make_ctx(), tamper=old_date)
    assert "stale_evidence" in codes(rep)

    def old_config(runner):
        runner.records["tc_003"]["policy_hash"] = "0" * 64
    rep2, *_ = validate(make_ctx(), tamper=old_config)
    assert "stale_evidence" in codes(rep2)


def test_evidence_from_a_failed_tool_call_is_not_evidence():
    ctx = make_ctx()
    s = Script(ctx, extra_calls=[("plan_experiment", {"channel": "applovin", "new_budget": 110.0}), ("get_policy_decision", {"channel": "nope"})])
    runner = T.ToolRunner(ctx, 50)
    for name, args in s.calls:
        runner.execute(ToolCall(name, args))
    items = base_items(s)
    items[0]["claims"].append(claim("tc_%03d" % len(s.calls), "action", "hold", "policy_decision", "meta"))
    out, _ = V.parse_output(output(items))
    assert "tool_call_failed" in codes(V.validate_agent_output(out, ctx, runner))


def test_every_item_must_cite_the_policy_decision_for_its_channel():
    rep, *_ = validate(make_ctx(), lambda items, s: item(items, "unity").__setitem__("claims", []))
    assert "missing_required_evidence" in codes(rep)
    rep2, *_ = validate(make_ctx(), lambda items, s: item(items, "unity")["claims"][0].__setitem__("value", "hold"))  # cites a different action than the tool returned
    assert "value_mismatch" in codes(rep2)


def test_numbers_written_in_prose_must_be_cited_values():
    uncited = lambda items, s: item(items, "meta").__setitem__("reason", "Policy action hold; expected ROAS is 0.83 and retention is 41%.")
    rep, *_ = validate(make_ctx(), uncited)
    assert [e["detail"] for e in rep["errors"] if e["code"] == "uncited_number"].__len__() == 2
    fine = lambda items, s: item(items, "applovin").__setitem__("reason", "Raise from 100 to 110 (a 10% step, 10 per day); lower bound 0.62 vs target 0.50 on 2025-10-13; D30 net revenue.")
    rep2, *_ = validate(make_ctx(), fine)
    assert rep2["ok"], rep2["errors"]  # budgets, the step, its percentage, cited values (0.50 = 0.5 target), a date and 'D30' are all fine
    summary_bad, *_ = validate(make_ctx(), summary="We should expect 12.5 extra installs.")
    assert "uncited_number" in codes(summary_bad)


# ------------------------------------------------------------------------------------------------ permitted actions and budgets


@pytest.mark.parametrize("ch,action,budget,expected", [
    ("meta", "decrease", 81.0, "forbidden_substitution"),  # policy hold -> decrease
    ("meta", "increase_test", 99.0, "forbidden_substitution"),  # policy hold -> increase
    ("unity", "hold", 60.0, "forbidden_substitution"),  # policy insufficient_evidence -> hold
    ("unity", "decrease", 54.0, "forbidden_substitution"),
    ("google_uac", "increase_test", 88.0, "forbidden_substitution"),  # policy decrease -> increase
    ("google_uac", "decrease", 64.0, "cut_larger_than_policy"),  # deeper cut than the policy's 72
    ("applovin", "decrease", 90.0, "forbidden_substitution"),  # policy increase -> decrease
])
def test_forbidden_actions_are_rejected(ch, action, budget, expected):
    def change(items, s):
        i = item(items, ch)
        i.update(action=action, new_budget=budget)
    rep, *_ = validate(make_ctx(), change)
    assert expected in codes(rep) and not rep["ok"]


def test_more_cautious_permitted_substitutions_are_accepted():
    def cautious(items, s):
        item(items, "applovin").update(action="hold", new_budget=100.0, qualification="none", experiment_plan_hash=None)
        item(items, "google_uac").update(action="hold", new_budget=80.0)
        item(items, "meta").update(action="insufficient_evidence", new_budget=90.0)
        item(items, "applovin")["claims"] = item(items, "applovin")["claims"][:1]
        item(items, "applovin")["reason"] = "Policy proposes an increase but I hold."
    rep, *_ = validate(make_ctx(), cautious)
    assert rep["ok"], rep["errors"]
    smaller_cut = lambda items, s: item(items, "google_uac").update(new_budget=76.0)
    rep2, *_ = validate(make_ctx(), smaller_cut)
    assert rep2["ok"], rep2["errors"]  # a decrease no larger than the policy's is permitted


def test_hold_and_insufficient_evidence_must_keep_the_budget_and_actions_must_match_the_budget_change():
    rep, *_ = validate(make_ctx(), lambda items, s: item(items, "meta").update(new_budget=95.0))
    assert "budget_changed_without_action" in codes(rep)
    rep2, *_ = validate(make_ctx(), lambda items, s: item(items, "google_uac").update(new_budget=85.0))  # a 'decrease' that increases
    assert "action_budget_mismatch" in codes(rep2)
    rep3, *_ = validate(make_ctx(), lambda items, s: item(items, "unity").update(new_budget=-5.0))
    assert "invalid_budget" in codes(rep3)
    rep4, *_ = validate(make_ctx(), lambda items, s: item(items, "meta").update(qualification="experiment_backed", experiment_plan_hash="a" * 64))
    assert "unexpected_plan" in codes(rep4)


def test_the_proposal_must_cover_each_paid_channel_exactly_once():
    rep, *_ = validate(make_ctx(), lambda items, s: items.pop() and None)
    assert codes(rep) == ["channel_set_mismatch"]
    rep2, *_ = validate(make_ctx(), lambda items, s: items.append(copy.deepcopy(items[0])))
    assert codes(rep2) == ["channel_set_mismatch"]


def test_hard_constraints_and_reversibility_are_enforced_on_the_whole_proposal():
    # the ceiling leaves no room: 400 + 10 + (-8) > 405
    tight = make_ctx(ceiling=401.0)
    rep, *_ = validate(tight, lambda items, s: item(items, "google_uac").update(action="hold", new_budget=80.0, claims=item(items, "google_uac")["claims"]))
    assert "constraint_violation" in codes(rep) and any("ceiling" in e["detail"] for e in rep["errors"])
    capped = make_ctx()
    capped.pcfg["constraints"]["per_channel_change_cap"] = 0.05  # the 10% step now breaks the cap, and so would its inverse
    rep2, *_ = validate(capped)
    assert "constraint_violation" in codes(rep2)


# ------------------------------------------------------------------------------------------------ increases and plans


def test_an_increase_without_its_plan_hash_or_with_the_wrong_one_is_not_actionable():
    def no_plan(items, s):
        item(items, "applovin").update(qualification="none", experiment_plan_hash=None)
    rep, *_ = validate(make_ctx(), no_plan)
    assert "increase_requires_bound_adequate_plan" in codes(rep)
    rep2, *_ = validate(make_ctx(), lambda items, s: item(items, "applovin").update(experiment_plan_hash="f" * 64))
    assert "increase_requires_bound_adequate_plan" in codes(rep2)
    other_plan = P.plan_hash(make_ctx().plan_for("applovin", 105.0))  # a real plan, but for a different step
    rep3, *_ = validate(make_ctx(), lambda items, s: item(items, "applovin").update(experiment_plan_hash=other_plan))
    assert "increase_requires_bound_adequate_plan" in codes(rep3)


def test_an_increase_whose_experiment_is_inadequate_cannot_be_proposed_and_the_reference_becomes_insufficient_evidence():
    ctx = make_ctx(adequate=False)
    assert ctx.reference("applovin")["action"] == "insufficient_evidence" and ctx.reference("applovin")["design_adjusted"] is True

    def ask_for_the_increase(items, s):  # a model that ignores the planning tool's verdict and asks for the policy's increase anyway
        tc = s.id("plan_experiment", "applovin", 110.0)
        item(items, "applovin").update(action="increase_test", new_budget=110.0, qualification="experiment_backed", experiment_plan_hash=P.plan_hash(ctx.plan_for("applovin", 110.0)))
        item(items, "applovin")["claims"] = [s.pol_claim("applovin"), claim(tc, "qualifies", True, "experiment_plan_for_exact_step", "applovin")]
        item(items, "applovin")["reason"] = "Raise from 100 to 110."
    rep, *_ = validate(ctx, ask_for_the_increase)
    assert "forbidden_substitution" in codes(rep) and "inadequate" in " ".join(e["detail"] for e in rep["errors"])
    assert "value_mismatch" in codes(rep)  # and the claim 'qualifies == true' contradicts the planner's output

    def abstain(items, s):
        item(items, "applovin").update(action="insufficient_evidence", new_budget=100.0, qualification="none", experiment_plan_hash=None)
        item(items, "applovin")["claims"] = item(items, "applovin")["claims"][:1]
        item(items, "applovin")["reason"] = "The experiment design is inadequate, so the increase cannot be tested."
    rep2, *_ = validate(ctx, abstain)
    assert rep2["ok"], rep2["errors"]
    hold = lambda items, s: abstain(items, s) or item(items, "applovin").update(action="hold")
    rep3, *_ = validate(ctx, hold)
    assert "forbidden_substitution" in codes(rep3)  # hold would say 'adequate evidence, no change justified': not permitted when the design is inadequate


def test_a_plan_the_model_never_asked_the_planner_for_cannot_be_cited():
    rep, *_ = validate(make_ctx(), lambda items, s: item(items, "applovin")["claims"].__setitem__(slice(1, None), []))
    assert "missing_plan_evidence" in codes(rep)


def test_a_smaller_step_needs_its_own_deterministic_recalculation_and_plan():
    ctx = make_ctx()
    s = Script(ctx, extra_calls=[("plan_experiment", {"channel": "applovin", "new_budget": 110.0}), ("plan_experiment", {"channel": "applovin", "new_budget": 105.0})])
    runner = T.ToolRunner(ctx, 50)
    for name, args in s.calls:
        runner.execute(ToolCall(name, args))
    items = base_items(s)
    a = item(items, "applovin")
    h105 = P.plan_hash(ctx.plan_for("applovin", 105.0))
    tc = s.id("plan_experiment", "applovin", 105.0)
    a.update(new_budget=105.0, experiment_plan_hash=h105)
    a["claims"] = [s.pol_claim("applovin"), claim(tc, "qualifies", True, "experiment_plan_for_exact_step", "applovin")]
    a["reason"] = "A smaller step from 100 to 105 that still qualifies."
    out, _ = V.parse_output(output(items))
    assert V.validate_agent_output(out, ctx, runner)["ok"]
    # the plan for the policy's step does not qualify a different step
    a["experiment_plan_hash"] = P.plan_hash(ctx.plan_for("applovin", 110.0))
    out, _ = V.parse_output(output(items))
    assert "increase_requires_bound_adequate_plan" in codes(V.validate_agent_output(out, ctx, runner))
    # a step whose recalculated lower bound is below the target is refused even with a plan, a hash and citations
    weak = make_ctx(lower_at=lambda step: 0.62 if step >= 110.0 else 0.4)
    runner2 = T.ToolRunner(weak, 50)
    for name, args in s.calls:
        runner2.execute(ToolCall(name, args))
    a["experiment_plan_hash"] = P.plan_hash(weak.plan_for("applovin", 105.0))
    out, _ = V.parse_output(output(items))
    rep = V.validate_agent_output(out, weak, runner2)
    assert "increase_not_qualified" in codes(rep) and "below the target" in " ".join(e["detail"] for e in rep["errors"])
    # a step below 2% of the budget is not proposed at all
    a.update(new_budget=101.0)
    out, _ = V.parse_output(output(items))
    assert "step_below_minimum" in codes(V.validate_agent_output(out, ctx, runner))


def test_a_step_larger_than_the_policy_step_is_refused():
    ctx = make_ctx()
    s = Script(ctx, extra_calls=[("plan_experiment", {"channel": "applovin", "new_budget": 110.0}), ("plan_experiment", {"channel": "applovin", "new_budget": 115.0})])
    runner = T.ToolRunner(ctx, 50)
    for name, args in s.calls:
        runner.execute(ToolCall(name, args))
    items = base_items(s)
    items[0].update(new_budget=115.0, experiment_plan_hash=P.plan_hash(ctx.plan_for("applovin", 115.0)))
    out, _ = V.parse_output(output(items))
    assert "step_larger_than_policy" in codes(V.validate_agent_output(out, ctx, runner))


def test_a_plan_that_is_not_bound_to_the_review_is_refused_even_if_the_planner_returned_it():
    wrong_policy = make_ctx(planner=lambda ch, new: __import__("tests.experiment_helpers", fromlist=["x"]).manual_plan(
        channel=ch, baseline=100.0, proposed=new, policy_hash="e" * 64, as_of=AS_OF, target=0.5))
    rep, *_ = validate(wrong_policy)
    assert "plan_invalid_or_unbound" in codes(rep) and "different policy configuration" in " ".join(e["detail"] for e in rep["errors"])


# ------------------------------------------------------------------------------------------------ malformed citations never crash and are never evidence


def runner_with(ctx, *calls):
    r = T.ToolRunner(ctx, 50)
    for name, args in calls:
        r.execute(ToolCall(name, args))
    return r


def check(ctx, runner, tc, field, value, population, channel="meta"):
    return V._check_claim(V.Claim(tool_call_id=tc, field=field, value=value, population=population, channel=channel), channel, ctx, runner)


def test_citing_a_null_object_or_list_field_is_a_structured_error_not_an_exception():
    """REGRESSION (independent review): get_economics for meta has response_curve = null; citing it with the number 0 used to raise TypeError out of run_agent."""
    ctx = make_ctx()
    r = runner_with(ctx, ("get_economics", {"channel": "meta"}), ("get_quality_checks", {"channel": "meta"}), ("get_economics", {"channel": "applovin"}))
    assert r.records["tc_001"]["result"]["response_curve"] is None
    for tc, field, pop, kind in (("tc_001", "response_curve", "historical_spend_install_response", "null"), ("tc_001", "marginal_step", "included_cohorts_model_estimated_finite_step", "null"),
                                 ("tc_001", "payback_target", "mature_history_payback_target", "object"), ("tc_003", "response_curve", "historical_spend_install_response", "object"),
                                 ("tc_002", "stages", "weekly_window_quality_checks", "list"), ("tc_003", "response_curve.spend_range", "historical_spend_install_response", "list"),
                                 ("tc_003", "marginal_step.assumptions", "included_cohorts_model_estimated_finite_step", "list")):
        for value in (0, 0.5, "x", True):
            errs = check(ctx, r, tc, field, value, pop, "applovin" if tc == "tc_003" else "meta")
            assert [e["code"] for e in errs] == ["unsupported_field_type"] and kind in errs[0]["detail"], (tc, field, value)


def test_incompatible_value_types_are_refused_and_nothing_is_converted_to_make_them_match():
    ctx = make_ctx()
    r = runner_with(ctx, ("get_policy_decision", {"channel": "meta"}), ("get_economics", {"channel": "applovin"}), ("plan_experiment", {"channel": "applovin", "new_budget": 110.0}))
    pol = "policy_decision"
    for value in (0.0, 1.0, True):
        assert [e["code"] for e in check(ctx, r, "tc_001", "action", value, pol)] == ["value_type_mismatch"]  # text field, non-text claim
    for value in ("0.5", "0.527", True, False):
        assert [e["code"] for e in check(ctx, r, "tc_002", "payback_target.target_d30_roas", value, "mature_history_payback_target", "applovin")] == ["value_type_mismatch"]  # number field
    for value in (1.0, 1, "true", "True"):
        assert [e["code"] for e in check(ctx, r, "tc_003", "qualifies", value, "experiment_plan_for_exact_step", "applovin")] == ["value_type_mismatch"]  # bool field: 1.0 is not True
    assert [e["code"] for e in check(ctx, r, "tc_002", "payback_target.target_d30_roas", float("nan"), "mature_history_payback_target", "applovin")] == ["value_type_mismatch"]
    assert [e["code"] for e in check(ctx, r, "tc_002", "payback_target.target_d30_roas", float("inf"), "mature_history_payback_target", "applovin")] == ["value_type_mismatch"]
    # controls: correctly typed scalar citations still work
    assert check(ctx, r, "tc_001", "action", "hold", pol) == []
    assert check(ctx, r, "tc_002", "payback_target.target_d30_roas", 0.5, "mature_history_payback_target", "applovin") == []
    assert check(ctx, r, "tc_003", "qualifies", True, "experiment_plan_for_exact_step", "applovin") == []
    assert [e["code"] for e in check(ctx, r, "tc_002", "payback_target.target_d30_roas", 0.9, "mature_history_payback_target", "applovin")] == ["value_mismatch"]


@pytest.mark.parametrize("claimed", [None, 0, 1.5, "x", True, [], {}, float("nan"), float("inf"), (1,), object()])
@pytest.mark.parametrize("actual", [None, 0, 1.5, "x", True, [1], {"a": 1}, float("nan"), float("-inf")])
def test_value_matches_is_total_it_never_raises(claimed, actual):
    assert isinstance(V.value_matches(claimed, actual), bool)


# ------------------------------------------------------------------------------------------------ signs, percentages and units in prose


def prose(reason, extra_calls=(), claims=None):
    """Validate a hold item for meta whose reason is `reason`, with the given extra (verified) claims added."""
    def edit(items, s):
        i = item(items, "meta")
        i["reason"] = reason
        i["claims"] += (claims(s) if claims else [])
    rep, *_ = validate(make_ctx(), edit, calls_extra=extra_calls)
    return rep


FORECAST = [("get_forecast_and_populations", {"channel": "meta"})]


def forecast_claims(s, *which):
    tc = "tc_%03d" % len(s.calls)
    table = {"revenue": ("included.forecast_d30_net_revenue", 300.0, "included_cohorts"), "installs": ("included.installs", 700, "included_cohorts"),
             "cohorts": ("included.n_cohorts", 5, "included_cohorts"),
             "share": ("included_spend_share", 0.7, "channel_review_window"), "spend": ("included.spend", 500.0, "included_cohorts")}
    return [claim(tc, *table[w], "meta") for w in which]


def uncited(rep):
    return sorted(e["detail"].split("'")[1] for e in rep["errors"] if e["code"] == "uncited_number")


def test_positive_evidence_rendered_as_negative_is_rejected_including_unicode_minus_signs():
    """REGRESSION (independent review): cite included.forecast_d30_net_revenue = 300; 'Forecast revenue is -300; forecast revenue is 300%.' was accepted."""
    c = lambda s: forecast_claims(s, "revenue")
    rep = prose("Forecast revenue is -300; forecast revenue is 300%.", FORECAST, c)
    assert not rep["ok"] and uncited(rep) == ["-300", "300%"]  # BOTH statements are wrong: the sign, and the unit
    for minus in ("-", "−", "–"):  # hyphen-minus, Unicode minus, en dash
        assert uncited(prose(f"Forecast revenue is {minus}300.", FORECAST, c)) == [f"{minus}300"], repr(minus)
    assert uncited(prose("Forecast revenue is 300.", FORECAST, c)) == []  # control
    assert uncited(prose("Forecast revenue is +300.", FORECAST, c)) == []  # an explicit plus on a positive value is fine


def test_currency_and_count_values_cannot_authorise_a_percentage_of_the_same_digits():
    c = lambda s: forecast_claims(s, "revenue", "installs", "spend")
    assert uncited(prose("Revenue 300%, installs 700%, spend 500%.", FORECAST, c)) == ["300%", "500%", "700%"]
    assert uncited(prose("Revenue 300, installs 700, spend 500.", FORECAST, c)) == []  # the plain numbers are fine
    # the case where the SCALE would otherwise line up: a count of 5 is not 500%, and a money amount of 3 would not be 300%
    five = lambda s: forecast_claims(s, "cohorts")
    assert uncited(prose("Five cohorts: 5 of them, i.e. 500%.", FORECAST, five)) == ["500%"] and uncited(prose("There are 5 cohorts.", FORECAST, five)) == []
    entries = [(3.0, "money", False), (3.0, "count", False), (3.0, "number", False), (0.3, "ratio", False)]
    for t in ("300%", "3%"):
        assert [e["detail"] for e in V._text_numbers_cited(t, entries[:3], "reason", None)], t  # money / count / plain number: no percentage at all
    assert V._text_numbers_cited("30%", entries[:3] + [(0.3, "ratio", False)], "reason", None) == []  # only the ratio value authorises 30%
    assert V._text_numbers_cited("3", entries[:3], "reason", None) == []
    assert uncited(prose("The budget of 90 is 90% of itself.", FORECAST, c)) == ["90%"]  # the item's own budget (money) is not a percentage either
    assert uncited(prose("The policy step is 100%.", FORECAST, c)) == ["100%"]  # meta's hold has no step; digits equal to nothing cited as a ratio


def test_percentages_use_the_ratio_scale_30_percent_is_0_30():
    c = lambda s: forecast_claims(s, "share")  # included_spend_share = 0.7
    assert uncited(prose("The included cohorts hold 70% of the window's spend.", FORECAST, c)) == []
    assert uncited(prose("The included cohorts hold 70.0 % of the spend and 0.7 as a share.", FORECAST, c)) == []
    for wrong in ("0.7%", "7%", "700%", "70 percent of 0.7%"):
        assert uncited(prose(f"The included cohorts hold {wrong} of the spend.", FORECAST, c)), wrong
    assert uncited(prose("The included cohorts hold 70 of the spend.", FORECAST, c)) == ["70"]  # a bare 70 is not 0.7 either
    assert uncited(prose("The included cohorts hold 70 percent of the spend.", FORECAST, c)) == []


def test_supported_rounding_and_valid_signed_values_still_pass():
    c = lambda s: forecast_claims(s, "share")
    assert uncited(prose("Share 70% (0.70).", FORECAST, c)) == []
    # a negative value: the check_budget_constraints reserve with a tight ceiling is -9
    tight = make_ctx(ceiling=401.0)
    s = Script(tight, extra_calls=[("plan_experiment", {"channel": "applovin", "new_budget": 110.0}), ("check_budget_constraints", {"changes": [{"channel": "applovin", "new_budget": 110.0}]})])
    runner = T.ToolRunner(tight, 50)
    for name, args in s.calls:
        runner.execute(ToolCall(name, args))
    assert runner.records["tc_008"]["result"]["reserve_after"] == -9.0 and runner.unit_of(runner.records["tc_008"], "reserve_after") == "money"
    neg = claim("tc_008", "reserve_after", -9.0, "hard_constraints_check", "meta")
    for text, bad in (("The reserve would be -9.", []), ("The reserve would be −9.", []), ("The reserve would be -9.00.", []), ("The reserve would be 9.", ["9"]),
                      ("The reserve would be +9.", ["+9"]), ("The reserve would be -9%.", ["-9%"]), ("A reserve of -10.", ["-10"])):
        errs = V._text_numbers_cited(text, [V._entry(V.Claim(**neg), runner)], "reason", "meta")
        assert sorted(e["detail"].split("'")[1] for e in errs) == sorted(bad), text


def test_the_items_own_arithmetic_keeps_its_direction_and_its_unit():
    ctx = make_ctx()

    def text_for(channel, new, text):
        return V._text_numbers_cited(text, V.derived_entries(ctx.budgets[channel], new), "reason", channel)

    up = lambda t: [e["detail"].split("'")[1] for e in text_for("applovin", 110.0, t)]  # 100 -> 110: step +10 money, +10% ratio
    assert up("Raise from 100 to 110, a step of 10 per day, +10 or 10%, up +10%.") == []
    assert up("A cut of 10.") == [] and up("-10 per day") == ["-10"] and up("-10%") == ["-10%"] and up("a 100% step") == ["100%"] and up("110%") == ["110%"] and up("a 1% step") == ["1%"]
    down = lambda t: [e["detail"].split("'")[1] for e in text_for("google_uac", 72.0, t)]  # 80 -> 72: step -8, -10%
    assert down("Cut from 80 to 72: -8 per day, -10%, a cut of 8 or 10%.") == []
    assert down("+8") == ["+8"] and down("+10%") == ["+10%"] and down("a step of 80%") == ["80%"]


def test_prose_checks_are_described_as_numerical_consistency_checks():
    doc = V._text_numbers_cited.__doc__
    assert "NUMERICAL CONSISTENCY CHECK" in doc and "does NOT establish that" in doc and "sentences are true" in doc


# ------------------------------------------------------------------------------------------------ parsing


def test_malformed_and_schema_invalid_output():
    assert V.parse_output(None)[1][0]["code"] == "malformed_output" and V.parse_output("I think we should hold.")[1][0]["code"] == "malformed_output"
    assert V.parse_output("{not json}")[1][0]["code"] == "malformed_output"
    bad = V.parse_output('{"summary": "x", "items": [{"channel": "meta", "action": "double", "new_budget": 1, "reason": "r", "claims": []}]}')
    assert bad[0] is None and bad[1][0]["code"] == "schema_invalid"
    extra = V.parse_output('{"summary": "x", "items": [], "approve": true}')
    assert extra[0] is None and extra[1][0]["code"] == "schema_invalid"  # unknown keys are refused (no smuggled instructions or fields)
    ok = V.parse_output('```json\n{"summary": "x", "items": []}\n```')
    assert ok[0] is not None  # a fenced block is accepted


