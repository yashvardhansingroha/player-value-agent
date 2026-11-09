"""Phase 5 on REAL dev-seed review data (synthetic data, SIMULATED experiments): the tools wrap the existing verified functions and add no number of their own.
Needs the generated dev data (data/ is gitignored: run scripts/generate_data.py and scripts/run_backtest.py first); skipped otherwise.
The model is the scripted MockLLM: this checks plumbing against real deterministic outputs, not any model."""
import json
from pathlib import Path

import pytest

from src.agent import agent as A
from src.agent import tools as T
from src.agent.llm import LLMError, MockLLM, ToolCall
from src.agent.scripted import scripted_llm

ROOT = Path(__file__).resolve().parents[1]
HAVE_DATA = all((ROOT / "data" / p).exists() for p in ("decision/seed_1/decision.db", "forecasts/seed_1/forecasts.db", "hidden/seed_1/hidden_truth.json"))
pytestmark = pytest.mark.skipif(not HAVE_DATA, reason="generated dev data not present (data/ is gitignored)")


@pytest.fixture(scope="module")
def envs():
    from scripts.run_agent import build_environment
    return {d: build_environment(d) for d in ("2025-10-13", "2025-11-10")}


def test_tool_outputs_are_the_policy_functions_outputs(envs):
    ctx = envs["2025-10-13"]
    runner = T.ToolRunner(ctx, 50)
    for ch in ctx.channels:
        d, out = ctx.decision(ch), runner.execute(ToolCall("get_policy_decision", {"channel": ch}))["result"]
        assert out["action"] == d["action"] and out["proposed_budget"] == d["proposed_budget"] and out["current_budget"] == d["current_budget"] and out["reasons"] == d["reasons"]
    econ = runner.execute(ToolCall("get_economics", {"channel": "applovin"}))["result"]
    ev = ctx.decision("applovin")["evidence"]
    assert econ["payback_target"]["target_d30_roas"] == ev["payback_target"]["target_d30_roas"] and econ["marginal_step"]["lower_bound"] == ev["marginal_d30_roas_step"]["lower_bound"]["lower"]
    assert runner.execute(ToolCall("get_quality_checks", {"channel": "applovin"}))["result"]["n_failed"] == sum(1 for s in ctx.decision("applovin")["stages"] if not s["passed"])


def test_the_agents_planning_tool_reproduces_the_reviewed_phase_4_plan_exactly(envs):
    ctx = envs["2025-10-13"]
    out = T.ToolRunner(ctx, 50).execute(ToolCall("plan_experiment", {"channel": "applovin", "new_budget": ctx.decision("applovin")["proposed_budget"]}))["result"]
    p4 = json.loads((ROOT / "reports" / "phase4_results.json").read_text(encoding="utf-8"))
    assert out["plan"]["plan_hash"] == p4["scale1"]["plan_hash"]  # same units, assignment, step, windows, power study and standards as the reviewed Phase 4 plan
    assert out["qualifies"] is False and out["plan"]["adequate"] is False and out["action_after_design"] == "insufficient_evidence"
    assert out["recalculation"]["lower_bound_meets_target"] is True  # the policy's economics were fine; it is the EXPERIMENT that is inadequate
    p56 = [r["P(keep)"] for r in p4["scale1"]["power_table"] if r["treatment days"] == 56 and r["true iROAS / target"] == 1.5][0]
    assert out["plan"]["power_at_alternative"] == p56 and out["plan"]["power_at_alternative"] < out["plan"]["target_power"]


def test_inadequate_experiment_case_the_validated_proposal_abstains_with_cited_power_numbers(envs):
    ctx = envs["2025-10-13"]
    run = A.run_agent(ctx, scripted_llm(ctx))
    assert run.status == "agent_proposal" and run.validation["ok"] and run.mode == "mock"
    app = {i.channel: i for i in run.proposal.items}["applovin"]
    assert app.action == "insufficient_evidence" and app.new_budget == app.current_budget and app.experiment_plan_hash is None
    assert any(l.startswith("INADEQUATE EXPERIMENT: applovin") for l in run.labels) and run.plans_to_register == []
    assert {"qualifies", "plan.power_at_alternative", "plan.target_power"} <= {c["field"] for c in app.evidence["claims"]}


def test_outage_case_the_proposal_abstains_for_the_channel_with_the_tracking_gap(envs):
    ctx = envs["2025-11-10"]
    assert ctx.decision("google_uac")["action"] == "insufficient_evidence" and any(s["stage"] == "quality:tracking_gap" and not s["passed"] for s in ctx.decision("google_uac")["stages"])
    run = A.run_agent(ctx, scripted_llm(ctx))
    g = {i.channel: i for i in run.proposal.items}["google_uac"]
    assert run.status == "agent_proposal" and g.action == "insufficient_evidence" and g.new_budget == g.current_budget
    assert any(c["field"].endswith(".passed") and c["value"] is False for c in g.evidence["claims"])  # it cites the failed tracking_gap stage
    assert any(l.startswith("MISSING EVIDENCE: google_uac") for l in run.labels)


def test_api_failure_on_real_data_falls_back_to_the_same_gated_proposal(envs):
    ctx = envs["2025-10-13"]
    run = A.run_agent(ctx, MockLLM([LLMError("simulated outage")]))
    assert run.status == "rules_only_fallback" and run.fallback_validation["ok"] and run.proposal.source == "rules_only"
    app = {i.channel: i for i in run.proposal.items}["applovin"]
    assert app.action == "insufficient_evidence" and app.qualification == "none"  # the fallback keeps the experiment-design gate: no un-planned increase
