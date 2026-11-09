"""Phase 5 tools: read-only, deterministic, strictly typed, and nothing the model can use to run code, read files, or touch the ledger.
Synthetic hand-made review (tests/agent_helpers.py); the real-data check is in test_agent_integration.py."""
import re
from pathlib import Path

import pytest

from src.agent import tools as T
from src.agent.llm import ToolCall
from tests.agent_helpers import AS_OF, POLICY_HASH, TARGET, make_ctx

ROOT = Path(__file__).resolve().parents[1]
AGENT_FILES = ["context.py", "tools.py", "validator.py", "agent.py", "planning.py"]


def run_tool(runner, name, **args):
    return runner.execute(ToolCall(name, args))


def test_the_registry_is_exactly_the_seven_read_only_tools_and_none_can_run_code_read_files_or_touch_the_ledger():
    assert sorted(T.REGISTRY) == sorted(["get_review_overview", "get_quality_checks", "get_forecast_and_populations", "get_economics", "get_policy_decision",
                                         "plan_experiment", "check_budget_constraints"])
    for spec in T.TOOLS:
        assert not any(w in spec.name.lower() for w in T.FORBIDDEN_CAPABILITIES), spec.name
    # the planning tool takes ONLY a channel and the exact proposed budget: there is no argument through which the model could supply a power number
    assert set(T.REGISTRY["plan_experiment"].parameters["properties"]) == {"channel", "new_budget"}
    assert {d["name"] for d in T.declarations()} == set(T.REGISTRY)


def test_agent_code_has_no_sql_shell_eval_file_access_or_ledger_execution():
    banned = re.compile(r"sqlite3|subprocess|os\.system|\beval\(|\bexec\(|\bopen\(|\bconn\b|\.cursor\(|executescript|read_sql|\.apply\(|\.approve\(|\.rollback\(|"
                        r"register_frozen_plan|reload_config|\bLedger\(|ledger\.ledger")
    for name in AGENT_FILES + ["llm.py"]:
        text = (ROOT / "src" / "agent" / name).read_text(encoding="utf-8")
        text = re.sub(r'""".*?"""', "", text, flags=re.DOTALL)  # docstrings may describe what is NOT done
        code = "\n".join(l for l in text.splitlines() if not l.strip().startswith("#"))
        hits = banned.findall(code)
        assert not hits, (name, hits)
    assert "src.ledger.ledger" not in (ROOT / "src" / "agent" / "agent.py").read_text(encoding="utf-8")


def test_every_call_gets_a_sequential_id_and_an_envelope_that_pins_the_date_configuration_and_populations():
    runner = T.ToolRunner(make_ctx(), max_calls=20)
    e1 = run_tool(runner, "get_review_overview")
    e2 = run_tool(runner, "get_policy_decision", channel="meta")
    assert (e1["tool_call_id"], e2["tool_call_id"]) == ("tc_001", "tc_002")
    for e in (e1, e2):
        assert e["as_of"] == AS_OF and e["policy_hash"] == POLICY_HASH and "result" in e and e["populations"]["default"]
    assert e2["channel"] == "meta" and e2["result"]["action"] == "hold" and e2["result"]["proposed_budget"] == 90.0
    assert runner.records["tc_002"] is e2 and runner.n == 2


def test_tool_outputs_are_the_deterministic_values_not_new_numbers():
    ctx = make_ctx()
    runner = T.ToolRunner(ctx, 20)
    ov = run_tool(runner, "get_review_overview")["result"]
    assert ov["ledger"] == {"version": 1, "total_budget": 400.0, "ceiling": 480.0, "reserve": 80.0}
    assert ov["channels"]["unity"]["failed_stages"] == ["quality:tracking_gap"] and ov["channels"]["applovin"]["policy_action"] == "increase_test"
    q = run_tool(runner, "get_quality_checks", channel="unity")["result"]
    assert q["n_failed"] == 1 and q["stages"][2]["blocks"] == "all->insufficient_evidence"
    econ = run_tool(runner, "get_economics", channel="applovin")["result"]
    assert econ["payback_target"]["target_d30_roas"] == TARGET and econ["marginal_step"]["lower_bound"] == 0.62 and econ["response_curve"]["usable"] is True
    assert run_tool(runner, "get_economics", channel="meta")["result"]["marginal_step"] is None  # the policy never evaluated an increase for meta


def test_included_and_excluded_cohorts_are_kept_apart_with_their_own_population_labels():
    runner = T.ToolRunner(make_ctx(), 20)
    rec = run_tool(runner, "get_forecast_and_populations", channel="meta")
    assert rec["result"]["included"]["installs"] == 700 and rec["result"]["excluded_withheld"]["installs"] == 313
    assert runner.population_of(rec, "included.installs") == "included_cohorts"
    assert runner.population_of(rec, "excluded_withheld.installs") == "excluded_withheld_cohorts"
    assert runner.population_of(rec, "uncertainty.lo") == "included_cohorts_weekly_forecast_interval"
    assert runner.population_of(rec, "window.0") == "channel_review_window"


def test_plan_experiment_reports_the_planning_study_and_recalculation_without_accepting_any_statistic_from_the_model():
    ctx = make_ctx()
    runner = T.ToolRunner(ctx, 20)
    out = run_tool(runner, "plan_experiment", channel="applovin", new_budget=110.0)["result"]
    assert out["qualifies"] is True and out["is_policy_step"] is True and out["action_after_design"] == "increase_test"
    assert out["recalculation"]["lower_bound"] == 0.62 and out["recalculation"]["lower_bound_meets_target"] is True
    plan = ctx.plan_for("applovin", 110.0)
    assert out["plan"]["plan_hash"] == __import__("src.experiments.plan", fromlist=["x"]).plan_hash(plan) and out["plan"]["adequate"] is True
    alt = [a for a in plan["power_assessment"]["per_duration"]["14"]["alternatives"] if a["multiple"] == 1.5][0]
    assert out["plan"]["power_at_alternative"] == alt["keep_rate"]  # the number the model sees is the planner's, copied
    bad = run_tool(runner, "plan_experiment", channel="applovin", new_budget=110.0, power=0.99)
    assert bad["error"]["code"] == "bad_arguments"  # an extra argument (a power claim) is refused outright
    inadequate = make_ctx(adequate=False)
    out2 = run_tool(T.ToolRunner(inadequate, 20), "plan_experiment", channel="applovin", new_budget=110.0)["result"]
    assert out2["qualifies"] is False and out2["action_after_design"] == "insufficient_evidence" and out2["plan"]["adequate"] is False
    assert any("inadequate" in p for p in out2["problems"]) and out2["plan"]["reasons"]


def test_a_changed_step_is_recalculated_for_exactly_that_step():
    ctx = make_ctx(lower_at=lambda s: 0.62 if s <= 105.0 else 0.4)
    r = T.ToolRunner(ctx, 20)
    small = run_tool(r, "plan_experiment", channel="applovin", new_budget=105.0)["result"]
    big = run_tool(r, "plan_experiment", channel="applovin", new_budget=110.0)["result"]
    assert small["qualifies"] is True and small["is_policy_step"] is False and small["plan"]["plan_hash"] != ctx.qualification("applovin", 110.0)["plan_hash"]
    assert big["qualifies"] is False and big["recalculation"]["lower_bound"] == 0.4 and big["action_after_design"] == "insufficient_evidence"
    not_inc = run_tool(r, "plan_experiment", channel="meta", new_budget=99.0)["result"]
    assert not_inc["qualifies"] is False and "cannot be recalculated" in not_inc["problems"][0] and not_inc["plan"] is None


def test_check_budget_constraints_uses_the_hard_constraints_and_the_inverse():
    r = T.ToolRunner(make_ctx(), 20)
    ok = run_tool(r, "check_budget_constraints", changes=[{"channel": "applovin", "new_budget": 110.0}])["result"]
    assert ok["ok"] is True and ok["reserve_after"] == 70.0
    bad = run_tool(r, "check_budget_constraints", changes=[{"channel": "applovin", "new_budget": 140.0}])["result"]
    assert bad["ok"] is False and {v["constraint"] for v in bad["violations"]} >= {"per_channel_change_cap"}


@pytest.mark.parametrize("name,args,code", [
    ("execute_sql", {"query": "select 1"}, "unknown_tool"), ("run_python", {"code": "1+1"}, "unknown_tool"), ("read_file", {"path": "x"}, "unknown_tool"),
    ("approve_proposal", {"proposal_id": "P"}, "unknown_tool"), ("apply_change", {"proposal_id": "P"}, "unknown_tool"), ("rollback_change", {"change_id": "C"}, "unknown_tool"),
    ("get_policy_decision", {}, "bad_arguments"), ("get_policy_decision", {"channel": 5}, "bad_arguments"), ("get_policy_decision", {"channel": "nope"}, "unknown_channel"),
    ("get_policy_decision", {"channel": "meta", "sql": "x"}, "bad_arguments"), ("plan_experiment", {"channel": "meta", "new_budget": "ten"}, "bad_arguments"),
    ("plan_experiment", {"channel": "meta", "new_budget": float("nan")}, "bad_arguments"), ("check_budget_constraints", {"changes": [{"channel": "meta"}]}, "bad_arguments"),
    ("check_budget_constraints", {"changes": [{"channel": "ghost", "new_budget": 5.0}]}, "unknown_channel"),
])
def test_forbidden_unknown_and_malformed_calls_are_refused_as_errors_and_still_count_against_the_limit(name, args, code):
    r = T.ToolRunner(make_ctx(), max_calls=3)
    env = r.execute(ToolCall(name, args))
    assert env["error"]["code"] == code and "result" not in env and r.n == 1


def test_a_failing_deterministic_tool_is_reported_as_an_error_not_as_a_number():
    def boom(channel, new_budget):
        raise RuntimeError("planner exploded")

    r = T.ToolRunner(make_ctx(planner=boom), 5)
    env = run_tool(r, "plan_experiment", channel="applovin", new_budget=110.0)
    assert env["error"]["code"] == "tool_failed" and "planner exploded" in env["error"]["message"] and "result" not in env


def test_resolving_a_field_path():
    r = T.ToolRunner(make_ctx(), 5)
    rec = run_tool(r, "get_quality_checks", channel="unity")
    assert T.resolve(rec, "stages.2.stage") == (True, "quality:tracking_gap")
    assert T.resolve(rec, "stages.9.stage")[0] is False and T.resolve(rec, "nothing.here")[0] is False
