"""Hand-off to the sandbox ledger. The AGENT never touches the ledger: here the harness side (the test) registers the plan and a "person" approves and
applies. This shows an accepted agent proposal is consumable by the real ledger gates, and that an increase is not actionable without its registered plan.
Synthetic fixtures; MockLLM."""
from src.agent import agent as A
from src.agent.llm import LLMError, MockLLM
from src.experiments import plan as P
from src.ledger.ledger import Ledger
from tests.agent_helpers import AS_OF, BUDGETS, FCFG, PCFG, make_ctx, mock_for


def new_ledger(tmp_path, name="l.db"):
    led = Ledger(tmp_path / name, PCFG, FCFG)
    led.initialize(BUDGETS, now=AS_OF)
    return led


def test_an_accepted_agent_proposal_with_a_qualified_increase_passes_the_ledger_gates_only_after_its_plan_is_registered(tmp_path):
    ctx = make_ctx()
    s, llm = mock_for(ctx)
    run = A.run_agent(ctx, llm)
    assert run.status == "agent_proposal" and run.proposal.ledger_version == 1
    led = new_ledger(tmp_path)
    assert run.proposal.policy_hash == led.policy_hash
    led.create_proposal(run.proposal, AS_OF)
    refused = led.approve(run.proposal.proposal_id, "reviewer", AS_OF)
    assert refused["ok"] is False and refused["code"] == "frozen_plan_not_registered" and led.head() == 1  # not actionable: the plan is not registered
    for h in run.plans_to_register:  # the harness (not the agent) registers the deterministic plan the validator named
        assert led.register_frozen_plan(ctx.all_plans()[h], "harness", AS_OF) == h
    assert led.approve(run.proposal.proposal_id, "reviewer", AS_OF)["ok"]
    applied = led.apply(run.proposal.proposal_id, AS_OF)
    assert applied["ok"] and applied["kind"] == "APPLY"  # a real experiment-backed increase, not MECHANICS_TEST_APPLY
    after = led.state()["budgets"]
    assert after["applovin"] == 110.0 and after["google_uac"] == 72.0 and after["unity"] == 60.0 and after["meta"] == 90.0


def test_the_rules_only_fallback_also_passes_the_ledger_gates(tmp_path):
    ctx = make_ctx()
    run = A.run_agent(ctx, MockLLM([LLMError("down")]))
    led = new_ledger(tmp_path)
    led.create_proposal(run.proposal, AS_OF)
    for h in run.plans_to_register:
        led.register_frozen_plan(ctx.all_plans()[h], "harness", AS_OF)
    assert led.approve(run.proposal.proposal_id, "reviewer", AS_OF)["ok"] and led.apply(run.proposal.proposal_id, AS_OF)["ok"]


def test_a_proposal_labelled_inadequate_experiment_changes_no_increase_and_has_nothing_to_register(tmp_path):
    ctx = make_ctx(adequate=False)
    s, llm = mock_for(ctx)
    run = A.run_agent(ctx, llm)
    assert run.status == "agent_proposal" and run.plans_to_register == []
    app = {i.channel: i for i in run.proposal.items}["applovin"]
    assert app.action == "insufficient_evidence" and app.new_budget == 100.0
    led = new_ledger(tmp_path)
    led.create_proposal(run.proposal, AS_OF)
    assert led.approve(run.proposal.proposal_id, "reviewer", AS_OF)["ok"] and led.apply(run.proposal.proposal_id, AS_OF)["ok"]
    assert led.state()["budgets"]["applovin"] == 100.0  # the only change is the decrease; no un-tested increase was ever possible


def test_even_a_hand_edited_agent_proposal_cannot_smuggle_an_unplanned_increase_past_the_ledger(tmp_path):
    ctx = make_ctx()
    s, llm = mock_for(ctx)
    run = A.run_agent(ctx, llm)
    forged = run.proposal.model_copy(deep=True)
    app = next(i for i in forged.items if i.channel == "applovin")
    app.experiment_plan_hash = P.plan_hash(ctx.plan_for("applovin", 105.0))  # a hash that is registered, but is the plan for another step
    led = new_ledger(tmp_path)
    led.register_frozen_plan(ctx.plan_for("applovin", 105.0), "harness", AS_OF)
    led.create_proposal(forged, AS_OF)
    r = led.approve(forged.proposal_id, "reviewer", AS_OF)
    assert r["ok"] is False and r["code"] == "plan_binding_mismatch"  # the ledger independently refuses it
