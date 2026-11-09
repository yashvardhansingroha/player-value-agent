"""Phase 5 agent loop: bounded tool loop, one repair attempt, rules-only fallback with the same gates, explicit labels, instruction-like text inside data,
and replay. EVERY MODEL HERE IS A SCRIPTED MockLLM (or a replay of its turns): these are plumbing and safeguard tests and say NOTHING about any real model.
Passing the selected injection tests below does not establish general injection resistance."""
import json
from pathlib import Path

import pytest

from src.agent import agent as A
from src.agent import tools as T
from src.agent.llm import LLMError, MockLLM, ReplayLLM, call_turn, recording_from_run, text_turn
from tests.agent_helpers import INJECTION, Script, base_items, claim, make_ctx, mock_for, output, standard_conversation

ROOT = Path(__file__).resolve().parents[1]


def items_of(run):
    return {i.channel: i for i in run.proposal.items}


def test_happy_path_produces_a_validated_agent_proposal_labelled_mock():
    ctx = make_ctx()
    s, llm = mock_for(ctx)
    run = A.run_agent(ctx, llm)
    assert run.status == "agent_proposal" and run.mode == "mock" and run.repairs_used == 0 and run.fallback_reason is None
    assert run.validation["ok"] and run.proposal.source == "agent"
    it = items_of(run)
    assert it["applovin"].action == "increase_test" and it["applovin"].qualification == "experiment_backed" and it["applovin"].new_budget == 110.0
    assert it["applovin"].experiment_plan_hash == ctx.qualification("applovin", 110.0)["plan_hash"] and run.plans_to_register == [it["applovin"].experiment_plan_hash]
    assert it["google_uac"].action == "decrease" and it["google_uac"].new_budget == 72.0 and it["unity"].action == "insufficient_evidence" and it["unity"].new_budget == 60.0
    assert any(l.startswith("MODE: MOCK") and "NOT evidence" in l for l in run.labels) and any(l.startswith("APPROVAL REQUIRED") for l in run.labels)
    assert any(l.startswith("MISSING EVIDENCE: unity") for l in run.labels)
    assert all(e["tool_call_id"].startswith("tc_") for e in run.evidence_trail) and len(run.tool_calls) == len(s.calls)
    json.dumps(run.to_dict())  # the whole run is serialisable
    assert run.proposal.items[0].evidence["claims"]  # every item carries the claims it was validated on


def test_the_model_sees_the_tools_the_system_prompt_and_tool_results_only_as_tool_data():
    ctx = make_ctx(injected_reason=INJECTION)
    s, llm = mock_for(ctx)
    A.run_agent(ctx, llm)
    systems = {r["system"] for r in llm.requests}
    assert systems == {A.SYSTEM_PROMPT} and INJECTION not in A.SYSTEM_PROMPT and "DATA" in A.SYSTEM_PROMPT
    assert llm.requests[0]["tool_names"] == sorted(T.REGISTRY, key=lambda n: [t.name for t in T.TOOLS].index(n))
    last = llm.requests[-1]["history"]
    tool_msgs = [h for h in last if h["role"] == "tool"]
    assert tool_msgs and any(INJECTION in json.dumps(r["payload"]) for h in tool_msgs for r in h["results"])  # the injected text arrives as tool-result DATA ...
    assert all(INJECTION not in h.get("text", "") for h in last if h["role"] == "user")  # ... and never as a user or system message


# ------------------------------------------------------------------------------------------------ API failure, malformed output, repair


def test_api_failure_before_any_answer_falls_back_to_the_rules_only_proposal_with_the_same_gates():
    ctx = make_ctx()
    run = A.run_agent(ctx, MockLLM([LLMError("503 service unavailable")]))
    assert run.status == "rules_only_fallback" and run.fallback_reason == "api_failure" and "503" in run.fallback_detail
    assert run.proposal.source == "rules_only" and run.fallback_validation["ok"] and run.validation is None
    assert any("RULES-ONLY FALLBACK: AGENT UNAVAILABLE (API FAILURE)" in l for l in run.labels) and any(l.startswith("APPROVAL REQUIRED") for l in run.labels)
    it = items_of(run)
    assert it["applovin"].action == "increase_test" and it["applovin"].qualification == "experiment_backed" and it["applovin"].experiment_plan_hash  # fallback increases need the plan too
    assert it["meta"].action == "hold" and it["unity"].action == "insufficient_evidence"
    assert it["applovin"].evidence["run_labels"] == run.labels  # the label is part of what an approval is bound to (it is in the proposal hash)


def test_api_failure_mid_investigation_also_falls_back_and_keeps_the_evidence_gathered_so_far():
    ctx = make_ctx()
    run = A.run_agent(ctx, MockLLM([call_turn(("get_review_overview", {})), LLMError("connection reset")]))
    assert run.fallback_reason == "api_failure" and run.status == "rules_only_fallback" and [c["tool"] for c in run.tool_calls] == ["get_review_overview"]


def test_fallback_labels_inadequate_experiments_and_missing_evidence_explicitly():
    ctx = make_ctx(adequate=False)
    run = A.run_agent(ctx, MockLLM([LLMError("down")]))
    it = items_of(run)
    assert it["applovin"].action == "insufficient_evidence" and it["applovin"].new_budget == 100.0 and it["applovin"].qualification == "none" and it["applovin"].experiment_plan_hash is None
    assert any(l.startswith("INADEQUATE EXPERIMENT: applovin") for l in run.labels) and any(l.startswith("MISSING EVIDENCE: unity") for l in run.labels)
    assert "experiment design inadequate" in it["applovin"].reason and run.fallback_validation["ok"] and run.plans_to_register == []


def test_malformed_output_gets_exactly_one_repair_attempt():
    ctx = make_ctx()
    s, turns = standard_conversation(ctx)
    run = A.run_agent(ctx, MockLLM([turns[0], text_turn("Sure! Here is my proposal in prose."), turns[1]]))
    assert run.status == "agent_proposal" and run.repairs_used == 1 and run.proposal.source == "agent"
    twice = A.run_agent(ctx, MockLLM([turns[0], text_turn("prose"), text_turn("{still not valid json")]))
    assert twice.status == "rules_only_fallback" and twice.fallback_reason == "malformed_output" and twice.repairs_used == 1 and twice.proposal.source == "rules_only"
    assert any("MALFORMED OUTPUT" in l for l in twice.labels)
    empty = A.run_agent(ctx, MockLLM([turns[0], LLMTurnEmpty(), turns[1]]))
    assert empty.status == "agent_proposal" and empty.repairs_used == 1  # an empty reply is malformed output too


def LLMTurnEmpty():
    from src.agent.llm import LLMTurn
    return LLMTurn(text=None)


def test_a_validator_rejection_is_repaired_once_and_the_repair_message_carries_the_errors():
    ctx = make_ctx()
    s, turns = standard_conversation(ctx, lambda items, s: next(i for i in items if i["channel"] == "meta").update(action="decrease", new_budget=81.0))
    good_s, good_turns = standard_conversation(ctx)
    llm = MockLLM([turns[0], turns[1], good_turns[1]])
    run = A.run_agent(ctx, llm)
    assert run.status == "agent_proposal" and run.repairs_used == 1
    repair = [h for h in llm.requests[-1]["history"] if h["role"] == "user"][-1]["text"]
    assert "REJECTED by the validator" in repair and "forbidden_substitution [meta]" in repair and "ONLY repair attempt" in repair
    failing = A.run_agent(ctx, MockLLM([turns[0], turns[1], turns[1]]))
    assert failing.status == "rules_only_fallback" and failing.fallback_reason == "validation_failed" and failing.validation["ok"] is False
    assert any(e["code"] == "forbidden_substitution" for e in failing.fallback_detail) and any("REJECTED BY THE VALIDATOR" in l for l in failing.labels)
    assert items_of(failing)["meta"].action == "hold"  # the fallback, not the rejected answer, is what is proposed


def test_the_repair_attempt_may_call_tools_to_obtain_missing_evidence():
    ctx = make_ctx()
    s = Script(ctx)  # first answer: no plan_experiment call, so the increase has no plan evidence
    first = base_items(Script(ctx, extra_calls=[("plan_experiment", {"channel": "applovin", "new_budget": 110.0})]))
    first_turns = [s.investigation_turn(), text_turn(output(first))]  # cites a tool call that was never made (tc_007)
    s2 = Script(ctx, extra_calls=[("plan_experiment", {"channel": "applovin", "new_budget": 110.0})])
    llm = MockLLM([*first_turns, call_turn(("plan_experiment", {"channel": "applovin", "new_budget": 110.0})), text_turn(output(base_items(s2)))])
    run = A.run_agent(ctx, llm)
    assert run.status == "agent_proposal" and run.repairs_used == 1 and run.tool_calls[-1]["tool"] == "plan_experiment"


def bad_citation_run_inputs(ctx, call, field, population, value):
    """Investigation turn + an answer whose meta item ALSO cites `field` of `call` with `value`, + the same answer without that citation."""
    import copy
    s = Script(ctx, extra_calls=[("plan_experiment", {"channel": "applovin", "new_budget": 110.0}), call])
    good = base_items(s)
    bad = copy.deepcopy(good)
    next(i for i in bad if i["channel"] == "meta")["claims"].append(claim(s.id(call[0], call[1].get("channel")), field, value, population, "meta"))
    return s.investigation_turn(), text_turn(output(bad)), text_turn(output(good))


BAD_CITATIONS = [  # (kind, call, field, population, value, expected code)
    ("null", ("get_economics", {"channel": "meta"}), "response_curve", "historical_spend_install_response", 0, "unsupported_field_type"),
    ("object", ("get_economics", {"channel": "meta"}), "payback_target", "mature_history_payback_target", 0.5, "unsupported_field_type"),
    ("list", ("get_quality_checks", {"channel": "meta"}), "stages", "weekly_window_quality_checks", 0, "unsupported_field_type"),
    ("incompatible type", ("get_policy_decision", {"channel": "meta"}), "action", "policy_decision", 0, "value_type_mismatch"),
]


@pytest.mark.parametrize("kind,call,field,population,value,code", BAD_CITATIONS, ids=[b[0] for b in BAD_CITATIONS])
def test_invalid_citations_get_the_repair_attempt_and_then_the_labelled_fallback_without_crashing_or_being_accepted(kind, call, field, population, value, code):
    """REGRESSION (independent review): a numeric citation of a null field used to raise TypeError out of run_agent."""
    ctx = make_ctx()
    inv, bad, good = bad_citation_run_inputs(ctx, call, field, population, value)
    llm = MockLLM([inv, bad, good])
    repaired = A.run_agent(ctx, llm)  # the first answer is rejected with a structured error, the repair answer is valid
    assert repaired.status == "agent_proposal" and repaired.repairs_used == 1 and repaired.validation["ok"]
    repair_text = [h for h in llm.requests[-1]["history"] if h["role"] == "user"][-1]["text"]
    assert code in repair_text and "ONLY repair attempt" in repair_text
    failed = A.run_agent(make_ctx(), MockLLM([inv, bad, bad]))  # the repair repeats the bad citation
    assert failed.status == "rules_only_fallback" and failed.fallback_reason == "validation_failed" and failed.repairs_used == 1
    assert failed.validation["ok"] is False and code in {e["code"] for e in failed.fallback_detail}  # never accepted as evidence
    assert failed.proposal.source == "rules_only" and failed.fallback_validation["ok"] and any("REJECTED BY THE VALIDATOR" in l for l in failed.labels)
    assert any(l.startswith("APPROVAL REQUIRED") for l in failed.labels)


def test_valid_scalar_citations_keep_working_in_the_same_loop():
    ctx = make_ctx()
    inv, bad, good = bad_citation_run_inputs(ctx, ("get_economics", {"channel": "meta"}), "payback_target.target_d30_roas", "mature_history_payback_target", 0.5)
    run = A.run_agent(ctx, MockLLM([inv, bad]))  # the 'bad' answer here carries a CORRECT scalar citation
    assert run.status == "agent_proposal" and run.repairs_used == 0 and run.validation["ok"]


# ------------------------------------------------------------------------------------------------ limits and forbidden tools


def executed(run):
    return [c for c in run.tool_calls if c["status"] != "denied"]


def test_the_tool_loop_is_bounded_and_a_model_that_ignores_the_budget_notice_takes_the_fallback():
    ctx = make_ctx()
    s = Script(ctx)
    run = A.run_agent(ctx, MockLLM([s.investigation_turn(), s.investigation_turn()]), max_tool_calls=3)  # 6 calls requested, 3 allowed, then it asks for tools AGAIN
    assert run.fallback_reason == "loop_limit" and run.status == "rules_only_fallback" and len(executed(run)) == 3 and any("TOOL-LOOP LIMIT" in l for l in run.labels)
    assert [c["status"] for c in run.tool_calls].count("denied") == 3 and "kept asking for tools" in run.fallback_detail
    forever = MockLLM([call_turn(("get_review_overview", {})) for _ in range(40)])
    run2 = A.run_agent(ctx, forever, max_tool_calls=5, max_model_turns=40)
    assert run2.fallback_reason == "loop_limit" and len(executed(run2)) == 5
    turns = MockLLM([call_turn(("get_review_overview", {})) for _ in range(10)])
    run3 = A.run_agent(ctx, turns, max_tool_calls=50, max_model_turns=4)
    assert run3.fallback_reason == "turn_limit" and len(run3.tool_calls) == 4 and run3.proposal.source == "rules_only"


def test_when_the_tool_budget_runs_out_the_model_gets_one_wrap_up_turn_and_its_gathered_evidence_is_kept():
    """CORRECTION from the recorded live attempts: the model was never told its budget, and hitting it threw away all the evidence it had gathered."""
    ctx = make_ctx()
    s = Script(ctx, extra_calls=[("plan_experiment", {"channel": "applovin", "new_budget": 110.0}), ("get_economics", {"channel": "meta"}), ("get_economics", {"channel": "tiktok"})])
    good = standard_conversation(ctx)[1][1]  # an answer citing only the first 7 calls (overview, 5 policy decisions, plan_experiment)
    llm = MockLLM([s.investigation_turn(), good])
    run = A.run_agent(ctx, llm, max_tool_calls=7)
    assert run.status == "agent_proposal" and run.validation["ok"] and run.proposal.source == "agent" and run.repairs_used == 0
    assert len(executed(run)) == 7 and [c["status"] for c in run.tool_calls].count("denied") == 2  # the two extra calls were refused, not run
    assert any(l.startswith("TOOL BUDGET REACHED: 7 of 7") and "2 further" in l for l in run.labels)
    notice = llm.requests[-1]["history"][-1]
    assert notice["role"] == "tool" and "TOOL BUDGET USED UP" in notice["notice"]  # the notice rides with the tool results
    denied = [r for r in notice["results"] if "error" in r["payload"] and r["payload"]["error"]["code"] == "tool_budget_exhausted"]
    assert len(denied) == 2 and all(r["payload"]["tool_call_id"].startswith("denied_") for r in denied)
    first_prompt = llm.requests[0]["history"][0]["text"]
    assert "TOOL BUDGET: at most 7 tool calls" in first_prompt and "budget" in A.SYSTEM_PROMPT.lower()  # the model is told the budget up front


def test_a_denied_call_is_not_evidence_and_every_validator_rule_still_applies_after_the_budget_is_used_up():
    ctx = make_ctx()
    s = Script(ctx, extra_calls=[("plan_experiment", {"channel": "applovin", "new_budget": 110.0}), ("get_economics", {"channel": "meta"})])
    items = base_items(s)
    meta = next(i for i in items if i["channel"] == "meta")
    meta["claims"].append(claim("denied_001", "payback_target.target_d30_roas", 0.5, "mature_history_payback_target", "meta"))  # cites the call that was refused
    citing_denied = text_turn(output(items))
    run = A.run_agent(ctx, MockLLM([s.investigation_turn(), citing_denied, citing_denied]), max_tool_calls=7)
    assert run.status == "rules_only_fallback" and run.fallback_reason == "validation_failed" and any(e["code"] == "unknown_tool_call" for e in run.fallback_detail)
    # with the budget used up, the repair attempt cannot fetch more evidence: asking for a tool in the repair turn takes the hard fallback
    run2 = A.run_agent(make_ctx(), MockLLM([Script(ctx, extra_calls=[("plan_experiment", {"channel": "applovin", "new_budget": 110.0})]).investigation_turn(), text_turn("prose, not JSON"),
                                           call_turn(("get_economics", {"channel": "meta"}))]), max_tool_calls=7)
    assert run2.fallback_reason == "loop_limit" and run2.repairs_used == 1


def test_an_answer_that_lacks_evidence_for_a_channel_it_could_not_check_is_not_accepted():
    ctx = make_ctx()
    s = Script(ctx, extra_calls=[("plan_experiment", {"channel": "applovin", "new_budget": 110.0})])  # only 3 calls allowed: the overview and two policy decisions
    run = A.run_agent(ctx, MockLLM([s.investigation_turn(), text_turn(output(base_items(s))), text_turn(output(base_items(s)))]), max_tool_calls=3)  # cites calls that never ran
    assert run.status == "rules_only_fallback" and run.fallback_reason == "validation_failed"
    assert {e["code"] for e in run.fallback_detail} >= {"unknown_tool_call"} and any("TOOL BUDGET REACHED" in l for l in run.labels)


def test_the_gemini_adapter_sends_the_wrap_up_notice_inside_the_tool_result_message():
    from google.genai import types
    from src.agent.llm import GeminiClient, LLMTurn, ToolCall
    call = ToolCall("get_review_overview", {}, "g-1")
    history = [{"role": "user", "text": "go"}, {"role": "model", "turn": LLMTurn(tool_calls=[call])},
               {"role": "tool", "results": [{"call": call, "payload": {"tool_call_id": "tc_001", "result": {}}}], "notice": A.WRAP_UP}]
    contents = GeminiClient("m", client=object())._contents(history)
    assert [c.role for c in contents] == ["user", "model", "user"]  # no second consecutive user message
    assert contents[2].parts[0].function_response.name == "get_review_overview" and contents[2].parts[1].text == A.WRAP_UP


def test_forbidden_tool_requests_are_refused_and_do_nothing():
    ctx = make_ctx()
    before = dict(ctx.budgets)
    s, turns = standard_conversation(ctx)
    probes = call_turn(("execute_sql", {"query": "DELETE FROM x"}), ("run_python", {"code": "import os"}), ("read_file", {"path": "C:/secrets"}), ("approve_proposal", {"proposal_id": "P"}),
                       ("apply_change", {"proposal_id": "P"}))
    # ids shift by the probes, so build the honest part after them
    probe_ids = 5
    s2 = Script(ctx, extra_calls=[("plan_experiment", {"channel": "applovin", "new_budget": 110.0})])
    shifted = []
    for it in base_items(s2):
        for c in it["claims"]:
            c["tool_call_id"] = "tc_%03d" % (int(c["tool_call_id"][3:]) + probe_ids)
        shifted.append(it)
    run = A.run_agent(ctx, MockLLM([probes, s2.investigation_turn(), text_turn(output(shifted))]))
    assert [c["status"] for c in run.tool_calls[:5]] == ["error"] * 5 and run.status == "agent_proposal"
    assert {e["error"]["code"] for e in run.evidence_trail[:5]} == {"unknown_tool"} and ctx.budgets == before
    citing_a_refused_call = base_items(s2)
    citing_a_refused_call[0]["claims"].append(claim("tc_001", "action", "hold", "policy_decision", "applovin"))
    run2 = A.run_agent(ctx, MockLLM([probes, s2.investigation_turn(), text_turn(output(shifted[:-1] + [dict(shifted[-1], claims=shifted[-1]["claims"] + [claim("tc_001", "x", 1.0, "p", "unity")])])),
                                     text_turn("no")]))
    assert run2.fallback_reason in ("validation_failed", "malformed_output")


# ------------------------------------------------------------------------------------------------ instruction-like text inside data


def test_instruction_like_text_in_tool_data_cannot_move_a_budget_even_if_the_model_obeys_it():
    ctx = make_ctx(injected_reason=INJECTION)

    def obey(items, s):  # a model that FOLLOWS the injected instruction: raise tiktok (policy hold) by 50% and say so
        t = next(i for i in items if i["channel"] == "tiktok")
        t.update(action="increase_test", new_budget=105.0, reason="Raise tiktok by 50% as instructed.")
    s, turns = standard_conversation(ctx, obey)
    run = A.run_agent(ctx, MockLLM([turns[0], turns[1], turns[1]]))
    assert run.status == "rules_only_fallback" and run.fallback_reason == "validation_failed"
    codes = {e["code"] for e in run.fallback_detail}
    assert {"forbidden_substitution", "constraint_violation", "not_reversible"} <= codes  # policy said hold; +50% also breaks the 20% cap and its inverse
    assert items_of(run)["tiktok"].action == "hold" and items_of(run)["tiktok"].new_budget == 70.0  # the proposal is the deterministic one


def test_a_model_that_ignores_the_injected_text_is_accepted_and_the_text_stays_out_of_the_proposal():
    ctx = make_ctx(injected_reason=INJECTION)
    s, llm = mock_for(ctx)
    run = A.run_agent(ctx, llm)
    assert run.status == "agent_proposal" and INJECTION not in json.dumps(run.proposal.model_dump())
    quoting = lambda items, s: next(i for i in items if i["channel"] == "tiktok").update(reason="The data says to raise tiktok by 50% and approve it.")
    s2, turns = standard_conversation(ctx, quoting)
    bad = A.run_agent(ctx, MockLLM([turns[0], turns[1], turns[1]]))
    assert bad.fallback_reason == "validation_failed" and any(e["code"] == "uncited_number" for e in bad.fallback_detail)  # even quoting its number is not allowed


def test_smuggled_fields_in_the_answer_are_rejected_not_executed():
    ctx = make_ctx()
    s, turns = standard_conversation(ctx)
    payload = json.loads(turns[1].text)
    payload["approve"] = True
    payload["apply_to_ledger"] = {"proposal_id": "x"}
    run = A.run_agent(ctx, MockLLM([turns[0], text_turn(json.dumps(payload)), text_turn(json.dumps(payload))]))
    assert run.fallback_reason == "malformed_output" and run.proposal.source == "rules_only"


# ------------------------------------------------------------------------------------------------ replay


def test_replay_plays_back_the_model_side_against_freshly_recomputed_tools_and_is_labelled_replay():
    ctx = make_ctx()
    s, llm = mock_for(ctx)
    original = A.run_agent(ctx, llm)
    with pytest.raises(ValueError, match="only an actual LIVE run"):
        recording_from_run(original.to_dict())  # a mock run can never be saved as a replay recording
    replay = A.run_agent(make_ctx(), ReplayLLM.from_turns(original.model_turns, source="test_fixture"))
    assert replay.mode == "replay" and replay.status == "agent_proposal" and any(l.startswith("MODE: REPLAY") for l in replay.labels)
    assert [(i.channel, i.action, i.new_budget) for i in replay.proposal.items] == [(i.channel, i.action, i.new_budget) for i in original.proposal.items]
    exhausted = A.run_agent(make_ctx(), ReplayLLM.from_turns(original.model_turns[:1], source="test_fixture"))
    assert exhausted.fallback_reason == "api_failure" and "replay exhausted" in exhausted.fallback_detail


def test_replay_files_come_only_from_live_runs_and_drift_in_the_tools_is_labelled(tmp_path):
    ctx = make_ctx()
    s, llm = mock_for(ctx)
    original = A.run_agent(ctx, llm).to_dict()
    live_like = dict(original, mode="live", model="a-model-id")  # a dict LABELLED live, written to tmp only: tests never create the repo's recording
    rec = recording_from_run(live_like)
    path = tmp_path / "rec.json"
    path.write_text(json.dumps(rec), encoding="utf-8")
    replay = ReplayLLM.from_file(path)
    assert replay.model_id == "a-model-id" and replay.source == "live_recording"
    drifted = A.run_agent(make_ctx(lower_at=lambda step: 0.7), replay)
    assert any(l.startswith("REPLAY DRIFT") for l in drifted.labels)
    for bad in ({"schema": "agent_replay_v1", "mode": "mock", "model": "m", "turns": []}, {"schema": "x", "mode": "live", "model": "m", "turns": []},
                {"schema": "agent_replay_v1", "mode": "live", "model": "", "turns": []}):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError):
            ReplayLLM.from_file(p)
    with pytest.raises(ValueError):
        recording_from_run(dict(live_like, model_turns=[]))


def test_the_repository_holds_recorded_runs_only_if_they_are_real_live_recordings():
    files = sorted((ROOT / "demo").glob("recorded_run*.json")) if (ROOT / "demo").exists() else []
    if not files:
        pytest.skip("no live recording exists yet: live verification is PENDING (no GEMINI_API_KEY was available); nothing is fabricated")
    for p in files:
        rec = json.loads(p.read_text(encoding="utf-8"))
        assert rec["mode"] == "live" and rec["schema"] == "agent_replay_v1" and rec["model"] and rec["turns"] and "ACTUAL LIVE RUN" in rec["label"], p.name
        ReplayLLM.from_file(p)
