"""Two-stage Groq flow: investigation with local tools, then ONE tool-free request constrained by a strict JSON schema. OFFLINE: faked transport, fake clock, no network, no key.
ALL DATA IS SYNTHETIC. A faked exchange proves our code's behaviour, nothing about the real service or model. A schema-conforming answer proves SHAPE only: every test below that
matters for correctness checks that the UNCHANGED validator still rejects wrong populations, missing evidence, fabricated numbers and forbidden actions."""
import copy
import json
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, ConfigDict
from typing import Literal, Optional

from evals import freeze as F
from src.agent import agent as A
from src.agent import evidence_index as EI
from src.agent import groq_client as G
from src.agent import strict_schema as S
from src.agent import tools as T
from src.agent.groq_client import GroqClient, GroqError, HttpResult
from src.agent.llm import GeminiClient, LLMTurn, MockLLM, ToolCall, text_turn
from tests.agent_helpers import make_ctx, standard_conversation
from tests.test_groq_client import BUCKET, Clock, Fake, err, from_turn, resp

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config" / "agent_groq.yaml").read_text(encoding="utf-8"))
MODEL = CFG["model"]
STAGED = {"tpm": 100000, "staged_finalization": True}


def sclient(items, cfg=None):
    ck = Clock()
    return GroqClient(MODEL, {**CFG, **STAGED, **(cfg or {})}, transport=Fake(items), clock=ck.now, sleep=ck.sleep), ck


def structured(answer, usage=None, finish="stop", extra=None):
    """A transport answer for the finalization request. Asserts the request really is a tool-free strict-schema request."""
    def make(body):
        assert "response_format" in body and "tools" not in body and "tool_choice" not in body and "parallel_tool_calls" not in body and body["stream"] is False
        text = answer if isinstance(answer, str) else json.dumps(answer)
        used = usage or (len(json.dumps(body["messages"]) + json.dumps(body["response_format"])) // 4, 300)
        return resp({"role": "assistant", "content": text, **(extra or {})}, usage=used, finish=finish)
    return make


def done():
    return from_turn(text_turn("DONE"))


def staged_script(ctx, mutate=None):
    s, turns = standard_conversation(ctx, mutate)
    answer = json.loads(turns[1].text)
    return s, turns, answer


def run_staged(items, ctx=None, cfg=None, **kw):
    ctx = ctx or make_ctx()
    c, ck = sclient(items, cfg)
    return ctx, c, A.run_agent(ctx, c, max_tool_calls=kw.pop("max_tool_calls", 14), max_model_turns=kw.pop("max_model_turns", 20), **kw)


def happy_items(ctx):
    s, turns, answer = staged_script(ctx)
    return [from_turn(turns[0]), done(), structured(answer)], answer


# ---------------------------------------------------------------------------------------------- the schema adapter


def test_the_provider_schema_conforms_to_the_strict_subset_and_is_derived_from_the_contract():
    schema = S.provider_schema()
    assert S.check_strict(schema) == []
    from src.agent.validator import AgentItem, AgentOutput, Claim
    assert list(schema["properties"]) == list(AgentOutput.model_fields) and schema["required"] == list(AgentOutput.model_fields)  # every property required, none invented
    item = schema["properties"]["items"]["items"]
    assert list(item["properties"]) == list(AgentItem.model_fields) and item["required"] == list(AgentItem.model_fields)
    claim = item["properties"]["claims"]["items"]
    assert list(claim["properties"]) == list(Claim.model_fields) and claim["additionalProperties"] is False
    assert item["properties"]["action"]["enum"] == ["increase_test", "hold", "decrease", "insufficient_evidence"] and item["properties"]["qualification"]["enum"] == ["none", "experiment_backed"]
    assert item["properties"]["new_budget"]["type"] == "number"
    for nullable in (item["properties"]["experiment_plan_hash"], claim["properties"]["channel"]):  # defaults/optionals become nullable unions
        assert {b["type"] for b in nullable["anyOf"]} == {"string", "null"}
    assert {b["type"] for b in claim["properties"]["value"]["anyOf"]} == {"boolean", "number", "string"}
    assert "$ref" not in json.dumps(schema) and "default" not in json.dumps(schema).replace("description", "") and "title" not in json.dumps(schema)


def test_response_format_is_strict_json_schema():
    rf = S.response_format()
    assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True and rf["json_schema"]["name"] == S.SCHEMA_NAME and rf["json_schema"]["schema"] == S.provider_schema()


def test_the_adapter_makes_defaults_required_inlines_refs_and_drops_unsupported_keywords():
    class Inner(BaseModel):
        model_config = ConfigDict(extra="forbid")
        a: int = 3
        b: Optional[str] = None
        kind: Literal["x", "y"] = "x"

    class Outer(BaseModel):
        model_config = ConfigDict(extra="forbid")
        inner: Inner
        many: list[Inner] = []
        x: float = 1.5

    out = S.to_strict(Outer.model_json_schema())
    assert S.check_strict(out) == [] and out["required"] == ["inner", "many", "x"]
    assert out["properties"]["inner"]["required"] == ["a", "b", "kind"] and out["properties"]["inner"]["additionalProperties"] is False
    assert out["properties"]["inner"]["properties"]["b"]["anyOf"] == [{"type": "string"}, {"type": "null"}] and out["properties"]["inner"]["properties"]["kind"]["enum"] == ["x", "y"]
    assert out["properties"]["many"]["items"] == out["properties"]["inner"] and "$defs" not in out


def test_the_restriction_checker_names_each_violation():
    base = S.provider_schema()
    bad = copy.deepcopy(base)
    bad["additionalProperties"] = True
    bad["properties"]["summary"]["minLength"] = 3
    bad["required"].remove("summary")
    bad["properties"]["extra"] = {"$ref": "#/x"}
    text = " | ".join(S.check_strict(bad))
    assert "additionalProperties must be false" in text and "unsupported keyword(s) ['minLength']" in text and "every property must be required" in text and "$ref is not allowed" in text
    deep = {"type": "object", "properties": {"a": {"type": "array", "items": {"type": "object", "properties": {"b": {"type": "array", "items": {"type": "object", "properties": {"c": {"type": "array", "items": {"type": "object", "properties": {"d": {"type": "string"}}, "required": ["d"], "additionalProperties": False}}}, "required": ["c"], "additionalProperties": False}}}, "required": ["b"], "additionalProperties": False}}}, "required": ["a"], "additionalProperties": False}
    assert any("nested deeper" in p for p in S.check_strict(deep))
    assert any("no type" in p for p in S.check_strict({"type": "object", "properties": {"q": {}}, "required": ["q"], "additionalProperties": False}))


def test_the_validator_tools_and_context_are_unchanged_from_the_frozen_baseline():
    man = F.load_manifest()["frozen_files"]
    for f in ("src/agent/validator.py", "src/agent/tools.py", "src/agent/context.py", "src/policy/constraints.py"):
        assert F.file_hash(ROOT / f) == man[f], f"{f} must not change for structured finalization"


# ---------------------------------------------------------------------------------------------- the finalization request


def test_finalization_sends_a_strict_schema_and_no_tools_no_tool_choice_no_streaming():
    ctx = make_ctx()
    items, _ = happy_items(ctx)
    _, c, run = run_staged(items, ctx)
    assert run.status == "agent_proposal" and run.finalization == "staged_structured"
    inv = [b for b in c.transport.bodies if "response_format" not in b]
    fin = [b for b in c.transport.bodies if "response_format" in b]
    assert len(inv) == 2 and len(fin) == 1
    f = fin[0]
    assert f["response_format"] == S.response_format() and f["stream"] is False and f["model"] == "openai/gpt-oss-120b"
    for forbidden in ("tools", "tool_choice", "parallel_tool_calls", "functions", "function_call"):
        assert forbidden not in f
    assert not any(m["role"] == "tool" or m.get("tool_calls") for m in f["messages"]), "no tool messages: the evidence travels as a compact index"
    assert f["messages"][0]["content"].endswith(G.FINALIZATION_NOTE) and "NO tools are available" in f["messages"][0]["content"]
    assert f["reasoning_effort"] == "low" and f["temperature"] == 0.0 and f["max_completion_tokens"] > 0
    assert all("tools" in b and b["tool_choice"] == "auto" and "response_format" not in b for b in inv)  # investigation still uses local tools only
    assert inv[0]["messages"][0]["content"].endswith(G.STAGED_INVESTIGATION_NOTE) and "reply with the single word DONE" in inv[0]["messages"][0]["content"]


def test_investigation_and_finalization_share_the_same_authoritative_evidence():
    ctx = make_ctx()
    items, _ = happy_items(ctx)
    _, c, run = run_staged(items, ctx)
    fin_user = [b for b in c.transport.bodies if "response_format" in b][0]["messages"][1]["content"]
    # the index in the request is exactly the index of an equivalent evidence log built by the same deterministic tools
    runner = T.ToolRunner(ctx, 40)
    for rec in run.evidence_trail:
        runner.execute(ToolCall(rec["tool"], rec["args"]))
    assert EI.build(runner) in fin_user
    # every listed fact resolves to the validator's own record, with the validator's own population label
    assert runner.order == [r["tool_call_id"] for r in run.evidence_trail]
    for e in EI.entries(runner):
        found, value = T.resolve(runner.records[e["id"]], e["path"])
        assert found and value == e["value"] and runner.population_of(runner.records[e["id"]], e["path"]) == e["population"]
    assert f"[{runner.records['tc_002']['populations']['default']}]" in fin_user  # the exact label the validator expects is shown in brackets


def test_the_index_lists_failed_calls_as_such_and_never_as_evidence():
    ctx = make_ctx()
    runner = T.ToolRunner(ctx, 2)
    runner.execute(ToolCall("get_policy_decision", {"channel": "meta"}))
    runner.execute(ToolCall("get_policy_decision", {"channel": "nope"}))
    text = EI.build(runner)
    assert "tc_002 get_policy_decision({\"channel\":\"nope\"}) -> ERROR unknown_channel (a failed call is not evidence; do not cite it)" in text
    assert all(e["id"] == "tc_001" for e in EI.entries(runner))
    runner.deny(ToolCall("get_policy_decision", {"channel": "tiktok"}))  # a denied call has no record and no id the index could list
    assert "denied" not in EI.build(runner)


# ---------------------------------------------------------------------------------------------- stage 1: required evidence


def investigate(ctx, calls_per_turn):
    """A scripted investigation: each inner list is one model turn of (tool, args) calls."""
    return [from_turn(LLMTurn(tool_calls=[ToolCall(n, a, None) for n, a in batch])) for batch in calls_per_turn]


def policy_calls(ctx, channels=None):
    return [("get_policy_decision", {"channel": c}) for c in (channels or ctx.channels)]


PLAN = ("plan_experiment", {"channel": "applovin", "new_budget": 110.0})


def test_missing_required_evidence_is_requested_once_and_then_collected_by_the_model_itself():
    ctx = make_ctx()
    s, turns, answer = staged_script(ctx)
    first = [("get_review_overview", {})] + policy_calls(ctx, ["applovin", "google_uac", "meta", "tiktok"]) + [PLAN]  # unity is not queried yet: ids tc_001..tc_006, the plan is tc_006
    # the scripted answer numbers unity's call tc_006 and the plan tc_007; this run's order differs (plan tc_006, unity tc_007), so swap the two ids in the answer
    swap = {"tc_006": "tc_007", "tc_007": "tc_006"}
    text = json.dumps(answer)
    for old, new in swap.items():
        text = text.replace(f'"tool_call_id": "{old}"', f'"tool_call_id": "@{new}"')
    answer = json.loads(text.replace('"@', '"'))
    items = investigate(ctx, [first]) + [done(), from_turn(LLMTurn(tool_calls=[ToolCall("get_policy_decision", {"channel": "unity"}, None)])), done(), structured(answer)]
    _, c, run = run_staged(items, ctx)
    inv = run.stages["investigation"]
    assert inv["evidence_nudges"] == 1 and inv["required_evidence_complete"] is True and inv["ended_by"] == "model_done" and inv["missing_required_at_end"] == []
    nudge = [m for b in c.transport.bodies for m in b["messages"] if m["role"] == "user" and "REQUIRED EVIDENCE IS STILL MISSING" in m["content"]][0]
    assert 'get_policy_decision({"channel": "unity"})' in nudge["content"] and "plan_experiment" not in nudge["content"]
    assert [t["tool"] for t in run.tool_calls].count("get_policy_decision") == 5 and len(run.tool_calls) == 7  # the MODEL made every call; the harness added none
    assert run.status == "agent_proposal" and run.validation["ok"] and run.stages["finalization"]["validator_ok"] is True


def test_required_evidence_that_the_model_never_collects_takes_the_labelled_fallback_and_no_finalization_is_sent():
    ctx = make_ctx()
    first = [("get_review_overview", {})] + policy_calls(ctx, ["applovin", "google_uac"])
    items = investigate(ctx, [first]) + [done(), done(), done()]
    _, c, run = run_staged(items, ctx)
    assert run.status == "rules_only_fallback" and run.fallback_reason == "required_evidence_missing"
    assert {(m["tool"], m["args"]["channel"]) for m in run.fallback_detail["missing"]} == {("get_policy_decision", "meta"), ("get_policy_decision", "tiktok"), ("get_policy_decision", "unity"), ("plan_experiment", "applovin")}
    assert run.fallback_detail["nudges_used"] == 2
    assert not any("response_format" in b for b in c.transport.bodies), "finalization must not run without the required evidence"
    assert any("AGENT UNAVAILABLE (REQUIRED EVIDENCE MISSING)" in l for l in run.labels) and run.fallback_validation["ok"]
    assert run.stages["investigation"]["ended_by"] == "nudges_used_up" and run.stages["investigation"]["required_evidence_complete"] is False
    assert len(run.tool_calls) == 3  # nothing was called on the model's behalf


def test_the_experiment_qualification_is_required_for_a_retrieved_increase_at_the_policys_budget():
    ctx = make_ctx()
    everything_but_plan = [("get_review_overview", {})] + policy_calls(ctx)
    wrong_budget = ("plan_experiment", {"channel": "applovin", "new_budget": 105.0})
    items = investigate(ctx, [everything_but_plan, [wrong_budget]]) + [done(), done(), done(), done()]
    _, c, run = run_staged(items, ctx)
    assert run.fallback_reason == "required_evidence_missing" and run.fallback_detail["missing"] == [{"tool": "plan_experiment", "args": {"channel": "applovin", "new_budget": 110.0}}]
    ok_items = investigate(ctx, [everything_but_plan, [PLAN]]) + [done()]
    assert A.missing_required_evidence(ctx, _runner_after(ctx, everything_but_plan + [PLAN])) == []
    assert A.missing_required_evidence(ctx, _runner_after(ctx, everything_but_plan)) == [{"tool": "plan_experiment", "args": {"channel": "applovin", "new_budget": 110.0}}]
    assert A.missing_required_evidence(ctx, _runner_after(ctx, everything_but_plan + [wrong_budget])) != []  # a plan for another budget does not qualify THIS step
    assert ok_items


def _runner_after(ctx, calls):
    r = T.ToolRunner(ctx, 40)
    for n, a in calls:
        r.execute(ToolCall(n, a))
    return r


def test_a_failed_call_does_not_count_as_required_evidence():
    ctx = make_ctx()
    r = _runner_after(ctx, [("get_policy_decision", {"channel": "nope"})] + policy_calls(ctx, ["applovin", "google_uac", "meta", "tiktok"]))
    assert A.missing_required_evidence(ctx, r) == [{"tool": "get_policy_decision", "args": {"channel": "unity"}}, {"tool": "plan_experiment", "args": {"channel": "applovin", "new_budget": 110.0}}] or \
        {m["tool"] for m in A.missing_required_evidence(ctx, r)} == {"get_policy_decision", "plan_experiment"}


def test_an_exhausted_tool_budget_with_required_evidence_missing_is_the_labelled_fallback_not_a_waiver():
    ctx = make_ctx()
    items = investigate(ctx, [[("get_review_overview", {})] + policy_calls(ctx)]) + [done()]
    _, c, run = run_staged(items, ctx, max_tool_calls=3)
    assert run.fallback_reason == "required_evidence_missing" and run.stages["investigation"]["ended_by"] == "tool_budget_used_up"
    last_tool_turn = [m for m in c.transport.bodies[1]["messages"] if m["role"] == "user"][-1]
    assert last_tool_turn["content"] == A.STAGED_WRAP_UP and any("TOOL BUDGET REACHED" in l for l in run.labels)


def test_the_investigation_text_is_never_treated_as_an_answer():
    ctx = make_ctx()
    s, turns, answer = staged_script(ctx)
    items = [from_turn(turns[0]), from_turn(text_turn(json.dumps({"summary": "a complete-looking answer", "summary_claims": [], "items": []}))), structured(answer)]
    _, c, run = run_staged(items, ctx)
    assert run.status == "agent_proposal" and run.proposal.items and run.rejections == []  # the early JSON was ignored; the structured request produced the proposal


# ---------------------------------------------------------------------------------------------- semantic validation is unchanged


def run_with_answer(mutator):
    ctx = make_ctx()
    s, turns, answer = staged_script(ctx)
    bad = copy.deepcopy(answer)
    mutator(bad)
    items = [from_turn(turns[0]), done(), structured(bad), structured(bad)]
    _, c, run = run_staged(items, ctx)
    return c, run


def test_a_schema_conforming_answer_with_a_wrong_population_label_is_still_rejected():
    def m(a):
        a["items"][1]["claims"][0]["population"] = "default"
    c, run = run_with_answer(m)
    assert run.fallback_reason == "validation_failed" and run.repairs_used == 1 and any(e["code"] == "wrong_population" for r in run.rejections for e in r["errors"])


def test_missing_policy_evidence_in_the_answer_is_still_rejected():
    def m(a):
        a["items"][2]["claims"] = []
    c, run = run_with_answer(m)
    assert run.fallback_reason == "validation_failed" and any(e["code"] == "missing_required_evidence" for r in run.rejections for e in r["errors"])


def test_a_fabricated_number_in_the_answer_is_still_rejected():
    def m(a):
        for it in a["items"]:
            for cl in it["claims"]:
                if cl["field"] == "recalculation.lower_bound":
                    cl["value"] = 0.91
    c, run = run_with_answer(m)
    assert run.fallback_reason == "validation_failed" and any(e["code"] in ("wrong_value", "value_mismatch", "claim_value_mismatch") or "value" in e["code"] for r in run.rejections for e in r["errors"])


def test_a_forbidden_action_and_a_fabricated_call_id_are_still_rejected():
    def m(a):
        a["items"][2]["action"], a["items"][2]["new_budget"] = "increase_test", 120.0  # meta: policy says hold
    c, run = run_with_answer(m)
    assert run.fallback_reason == "validation_failed" and run.rejections

    def m2(a):
        a["items"][0]["claims"][0]["tool_call_id"] = "tc_099"
    c2, run2 = run_with_answer(m2)
    assert run2.fallback_reason == "validation_failed" and any("unknown" in e["code"] for r in run2.rejections for e in r["errors"])


def test_one_repair_is_allowed_is_recorded_and_can_succeed():
    ctx = make_ctx()
    s, turns, answer = staged_script(ctx)
    bad = copy.deepcopy(answer)
    bad["items"][1]["claims"][0]["population"] = "default"
    _, c, run = run_staged([from_turn(turns[0]), done(), structured(bad), structured(answer)], ctx)
    assert run.status == "agent_proposal" and run.repairs_used == 1 and len(run.rejections) == 1 and run.rejections[0]["stage"] == "finalization"
    fins = [b for b in c.transport.bodies if "response_format" in b]
    assert len(fins) == 2 and all("tools" not in b for b in fins)
    repair_msgs = fins[1]["messages"]
    assert repair_msgs[-2]["role"] == "assistant" and json.loads(repair_msgs[-2]["content"]) == bad and "REJECTED by the validator" in repair_msgs[-1]["content"] and "No tools are available" in repair_msgs[-1]["content"]
    st = run.stages["finalization"]
    assert st["requests"] == 2 and st["repairs_used"] == 1 and st["validator_ok"] is True and st["text_parsed_as_json_object"] is True
    # a second failure is NOT repaired again
    _, c2, run2 = run_staged([from_turn(turns[0]), done(), structured(bad), structured(bad), structured(answer)], ctx)
    assert run2.fallback_reason == "validation_failed" and run2.repairs_used == 1 and len([b for b in c2.transport.bodies if "response_format" in b]) == 2


def test_non_json_text_from_the_structured_request_is_a_malformed_output_rejection():
    ctx = make_ctx()
    s, turns, answer = staged_script(ctx)
    _, c, run = run_staged([from_turn(turns[0]), done(), structured("not json at all"), structured("{broken")], ctx)
    assert run.fallback_reason == "malformed_output" and [r["errors"][0]["code"] for r in run.rejections] == ["malformed_output"] * 2
    assert run.stages["finalization"]["text_parsed_as_json_object"] is False and c.request_log[-1]["content_is_json"] is False


# ---------------------------------------------------------------------------------------------- failures: provider, refusal, incomplete output, schema


@pytest.mark.parametrize("item,kind,label", [
    (err(400, "Failed to generate JSON. Please adjust your prompt. See 'failed_generation' for more details.", None), "schema_failure", "STRUCTURED OUTPUT FAILED"),
    (HttpResult(400, {}, json.dumps({"error": {"message": "Failed to generate JSON.", "type": "invalid_request_error", "code": "json_validate_failed", "failed_generation": "{}"}})), "schema_failure", "STRUCTURED OUTPUT FAILED"),
    (err(400, "response_format: json_schema is not supported for this model"), "bad_request", "REQUEST REJECTED (HTTP 400)"),
    (err(429, "Rate limit reached on tokens per day (TPD): Limit 200000."), "daily", "DAILY LIMIT REACHED"),
    (TimeoutError("slow"), "timeout", "TIMEOUT after the configured"),
    (HttpResult(200, {}, "garbage"), "malformed_response", "MALFORMED RESPONSE"),
    (lambda body: resp({"role": "assistant", "content": None, "refusal": "I can't help with that."}), "refusal", "MODEL REFUSED"),
    (lambda body: resp({"role": "assistant", "content": "{\"summary\": \"cut"}, finish="length"), "incomplete_output", "OUTPUT INCOMPLETE (finish_reason=length)"),
    (lambda body: resp({"role": "assistant", "content": ""}), "malformed_response", "no text content"),
    (lambda body: resp({"role": "assistant", "content": None, "tool_calls": [{"id": "x", "type": "function", "function": {"name": "get_review_overview", "arguments": "{}"}}]}), "malformed_response", "declared no tools"),
])
def test_a_finalization_failure_is_the_labelled_fallback_with_the_right_kind(item, kind, label):
    ctx = make_ctx()
    s, turns, answer = staged_script(ctx)
    _, c, run = run_staged([from_turn(turns[0]), done(), item, structured(answer)], ctx)
    assert run.status == "rules_only_fallback" and run.fallback_reason == "finalization_failed" and label in str(run.fallback_detail)
    assert run.stages["finalization"]["provider_accepted_schema_and_returned_text"] is False and run.stages["finalization"]["error_kind"] == kind
    assert run.fallback_validation["ok"] and any("AGENT UNAVAILABLE (STRUCTURED FINALIZATION FAILED)" in l for l in run.labels) and run.repairs_used == 0
    assert len([b for b in c.transport.bodies if "response_format" in b]) == 1, "a provider failure is not retried and not repaired"


def test_a_provider_error_during_investigation_is_an_api_failure_and_staged_runs_do_not_unwrap_answers():
    ctx = make_ctx()
    wrapped = HttpResult(400, BUCKET, json.dumps({"error": {"message": "Tool call validation failed: attempted to call tool 'JSON' which was not in request.tools", "type": "x", "code": "tool_use_failed",
                                                              "failed_generation": json.dumps({"name": "JSON", "arguments": {"summary": "s", "items": []}})}}))
    _, c, run = run_staged([wrapped], ctx)
    assert run.fallback_reason == "api_failure" and "MODEL TOOL-CALL FORMAT FAILURE" in str(run.fallback_detail) and c.salvaged == 0 and run.stages["investigation"]["ended_by"] == "api_failure"
    assert not any("response_format" in b for b in c.transport.bodies)


def test_an_oversized_finalization_prompt_is_refused_before_sending():
    ctx = make_ctx()
    s, turns, answer = staged_script(ctx)
    items = [from_turn(turns[0]), done(), structured(answer)]
    c, _ = sclient(items, cfg={"tpm": 3000})  # the investigation prompt alone (system, notes, tool schemas) would already be too big here
    run = A.run_agent(ctx, c, max_tool_calls=14)
    assert run.fallback_reason == "api_failure" and "REQUEST TOO LARGE (preflight)" in str(run.fallback_detail) and c.transport.bodies == []


# ---------------------------------------------------------------------------------------------- reporting by stage, labels, isolation


def test_each_stage_reports_its_own_outcome_and_tokens_and_they_add_up():
    ctx = make_ctx()
    items, _ = happy_items(ctx)
    _, c, run = run_staged(items, ctx)
    inv, fin = run.stages["investigation"], run.stages["finalization"]
    assert inv["model_requests"] == 2 and inv["tool_calls_by_model"] == 7 and inv["ended_by"] == "model_done" and inv["required_evidence_complete"] is True and inv["evidence_nudges"] == 0
    assert fin["requests"] == 1 and fin["provider_accepted_schema_and_returned_text"] is True and fin["text_parsed_as_json_object"] is True and fin["validator_ok"] is True and fin["repairs_used"] == 0
    assert inv["usage"]["prompt_tokens"] > 0 and fin["usage"]["prompt_tokens"] > 0
    assert run.usage == {"prompt_tokens": inv["usage"]["prompt_tokens"] + fin["usage"]["prompt_tokens"], "output_tokens": inv["usage"]["output_tokens"] + fin["usage"]["output_tokens"]}
    d = run.to_dict()
    assert d["finalization"] == "staged_structured" and d["stages"]["finalization"]["validator_ok"] is True and json.dumps(d)
    assert [t.get("stage") for t in run.model_turns] == ["investigation", "investigation", "finalization"]
    assert [r["stage"] for r in c.request_log] == ["investigation", "investigation", "finalization"] and "schema_sha256" in c.request_log[2]
    assert run.mode == "mock" and run.provider == "groq" and any(l.startswith("STAGED FINALIZATION:") and "SHAPE" in l for l in run.labels)
    assert any(l.startswith("MODE: MOCK (provider groq, model openai/gpt-oss-120b)") for l in run.labels)


def test_a_failed_finalization_reports_investigation_success_separately():
    ctx = make_ctx()
    s, turns, answer = staged_script(ctx)
    _, c, run = run_staged([from_turn(turns[0]), done(), err(400, "Failed to generate JSON.")], ctx)
    assert run.stages["investigation"]["required_evidence_complete"] is True and run.stages["investigation"]["ended_by"] == "model_done"
    assert run.stages["finalization"]["provider_accepted_schema_and_returned_text"] is False and run.stages["finalization"]["validator_ok"] is None


def test_gemini_mock_and_single_stage_behaviour_is_intact():
    ctx = make_ctx()
    run = A.run_agent(ctx, MockLLM(standard_conversation(ctx)[1]))
    assert run.finalization is None and run.stages == {} and run.status == "agent_proposal" and not any("STAGED" in l for l in run.labels)
    assert not hasattr(GeminiClient, "staged") and not hasattr(MockLLM([]), "staged") and not hasattr(MockLLM([]), "generate_structured")
    c, _ = sclient([], cfg={"staged_finalization": False})
    assert c.staged is False and c.investigation_note() == G.PROVIDER_NOTE
    d = run.to_dict()
    assert d["finalization"] is None and d["stages"] == {}


def test_the_shipped_staged_configuration_changes_only_the_finalization_and_keeps_the_isolation_settings():
    assert CFG["staged_finalization"] is True and CFG["max_evidence_nudges"] == 2
    assert CFG["reasoning_effort"] == "low" and CFG["model"] == "openai/gpt-oss-120b" and CFG["temperature"] == 0.0  # not changed together with the new stage
    assert "example" not in (G.STAGED_INVESTIGATION_NOTE + G.FINALIZATION_NOTE).lower()  # no examples added at the same time
    assert CFG["max_model_turns"] == 20 and CFG["max_tool_calls"] == 14 and CFG["repair_retries"] == 1  # the existing bounds
