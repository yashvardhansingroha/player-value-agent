"""The Groq provider adapter, tested OFFLINE with a faked transport and a fake clock: no network, no key, no real waiting. ALL DATA IS SYNTHETIC.
These tests show the adapter's serialisation, id matching, error handling, pacing and labelling are what they claim. They show NOTHING about how the real service
or the real model behaves: no live Groq capability has been demonstrated. An injected transport always reports mode 'mock'."""
import copy
import json
from pathlib import Path

import pytest
import yaml

from src.agent import agent as A
from src.agent import groq_client as G
from src.agent import tools as T
from src.agent.groq_client import GroqClient, GroqError, HttpResult, TokenPacer, classify_error, model_view, parse_duration
from src.agent.llm import LLMTurn, MockLLM, ToolCall, call_turn, recording_from_run, text_turn
from tests.agent_helpers import Script, base_items, claim, make_ctx, output, standard_conversation

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config" / "agent_groq.yaml").read_text(encoding="utf-8"))
MODEL = CFG["model"]


class Clock:
    def __init__(self):
        self.t, self.slept = 1000.0, []

    def now(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


class Fake:
    """Faked transport: items are HttpResult, an Exception to raise, or callable(body) -> HttpResult. Records every body and timeout it was handed."""

    def __init__(self, items):
        self.items, self.bodies, self.timeouts = list(items), [], []

    def __call__(self, body, timeout_seconds):
        self.bodies.append(copy.deepcopy(body))
        self.timeouts.append(timeout_seconds)
        it = self.items.pop(0)
        if isinstance(it, Exception):
            raise it
        return it(body) if callable(it) else it


def resp(message, usage=(900, 80), headers=None, finish="stop"):
    data = {"choices": [{"index": 0, "message": message, "finish_reason": finish}]}
    if usage:
        data["usage"] = {"prompt_tokens": usage[0], "completion_tokens": usage[1], "total_tokens": usage[0] + usage[1]}
    return HttpResult(200, headers or {}, json.dumps(data))


def err(status, message, headers=None):
    return HttpResult(status, headers or {}, json.dumps({"error": {"message": message, "type": "x"}}))


def from_turn(turn: LLMTurn):
    """A callable that answers in Groq's JSON shape for a neutral LLMTurn, with usage derived from the request size (so the pacing sees realistic numbers)."""
    def make(body):
        used = len(json.dumps(body["messages"]) + json.dumps(body["tools"])) // 3
        if turn.tool_calls:
            msg = {"role": "assistant", "content": None, "tool_calls": [{"id": f"call_{i}", "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.args)}}
                                                                        for i, c in enumerate(turn.tool_calls)]}
        else:
            msg = {"role": "assistant", "content": turn.text}
        return resp(msg, usage=(used, 120))
    return make


def client(items, cfg=None, clock=None, **kw):
    clock = clock or Clock()
    c = GroqClient(MODEL, {**CFG, "staged_finalization": False, **(cfg or {})}, transport=Fake(items), clock=clock.now, sleep=clock.sleep, **kw)  # the ORIGINAL single-stage flow unless a test asks for staged
    return c, clock


def plain_history(text="hi"):
    return [{"role": "user", "text": text}]


# ---------------------------------------------------------------------------------------------- durations and classification


@pytest.mark.parametrize("text,sec", [("7.66s", 7.66), ("2m59.56s", 179.56), ("1h2m", 3720.0), ("340ms", 0.34), ("1d1h", 90000.0), ("12", 12.0), ("1.5", 1.5), ("1m30s", 90.0)])
def test_durations_are_parsed(text, sec):
    assert parse_duration(text) == pytest.approx(sec)


@pytest.mark.parametrize("text", ["", "soon", "m", "5x"])
def test_unparseable_durations_are_none(text):
    assert parse_duration(text) is None


def test_errors_are_classified_by_status_and_wording():
    c = lambda status, msg, headers=None: classify_error(err(status, msg, headers))
    assert c(429, "Rate limit reached ... on tokens per minute (TPM): Limit 8000, Used 7000, Requested 3000. Please try again in 2.5s.").kind == "per_minute"
    assert c(429, "Rate limit reached ... on tokens per minute (TPM): Limit 8000. Please try again in 2.5s.").retry_after == 2.5
    assert c(429, "x", {"retry-after": "7"}).retry_after == 7.0
    assert c(429, "Rate limit reached ... on tokens per day (TPD): Limit 200000, Used 199000. Please try again in 9m30s.").kind == "daily"
    assert c(429, "Rate limit reached ... on requests per day (RPD): Limit 1000. Please try again in 14m24s.").kind == "daily"
    assert c(413, "Request too large for model on tokens per minute (TPM): Limit 8000, Requested 9500").kind == "request_too_large"
    assert c(401, "Invalid API Key").kind == "auth" and c(403, "forbidden").kind == "auth" and c(402, "payment required").kind == "billing"
    assert c(503, "over capacity").kind == "server" and c(400, "bad").kind == "bad_request"
    assert classify_error(HttpResult(429, {}, "not json at all")).kind == "per_minute"  # a non-JSON body is still classified, never a crash


# ---------------------------------------------------------------------------------------------- request body: local function calling only


def test_the_request_exposes_only_our_local_functions_and_no_hosted_tool():
    ctx = make_ctx()
    c, _ = client([])
    body = c.build_body(A.SYSTEM_PROMPT, [{"role": "user", "text": A.task_prompt(ctx, 14)}], T.declarations())
    assert [t["type"] for t in body["tools"]] == ["function"] * len(T.TOOLS) and [t["function"]["name"] for t in body["tools"]] == [t.name for t in T.TOOLS]
    assert body["tool_choice"] == "auto" and body["parallel_tool_calls"] is True and body["stream"] is False and body["model"] == "openai/gpt-oss-120b"
    flat = json.dumps(body).lower()
    for hosted in ("browser_search", "code_interpreter", "web_search", "compound", "search_settings", "mcp", "executed_tools"):
        assert hosted not in flat, hosted
    assert body["messages"][0]["role"] == "system" and body["messages"][0]["content"].startswith(A.SYSTEM_PROMPT) and body["messages"][0]["content"].endswith(G.PROVIDER_NOTE)
    assert body["reasoning_effort"] == "low" and body["max_completion_tokens"] > 0 and body["temperature"] == 0.0


def test_the_provider_note_does_not_weaken_a_rule_and_the_original_prompt_is_untouched():
    assert "approve" not in G.PROVIDER_NOTE.lower() and "ignore" not in G.PROVIDER_NOTE.lower()
    assert "get_policy_decision for ALL paid channels TOGETHER in ONE turn" in G.PROVIDER_NOTE and "validator" in G.PROVIDER_NOTE and "never as a tool call" in G.PROVIDER_NOTE
    import hashlib
    from evals import freeze as F
    assert hashlib.sha256(A.SYSTEM_PROMPT.encode()).hexdigest() == F.load_manifest()["agent_surface"]["system_prompt_sha256"]  # the frozen prompt text itself is unchanged
    assert hashlib.sha256(json.dumps(T.declarations(), sort_keys=True).encode()).hexdigest() == F.load_manifest()["agent_surface"]["tool_declarations_sha256"]


# ---------------------------------------------------------------------------------------------- message serialisation, parallel calls, id matching


def conversation(ctx, n_calls=3, ids=("a1", "b2", "c3")):
    calls = [ToolCall("get_policy_decision", {"channel": ch}, i) for ch, i in zip(ctx.channels, ids)][:n_calls]
    runner = T.ToolRunner(ctx, 40)
    results = [{"call": cl, "payload": runner.execute(cl)} for cl in calls]
    hist = [{"role": "user", "text": "start"}, {"role": "model", "turn": LLMTurn(tool_calls=calls)}, {"role": "tool", "results": results}]
    return hist, calls, results, runner


def test_parallel_tool_calls_keep_provider_ids_and_every_result_returns_with_its_own_id():
    ctx = make_ctx()
    hist, calls, results, _ = conversation(ctx)
    hist[2]["notice"] = A.WRAP_UP
    c, _ = client([])
    msgs = c.messages("SYS", hist)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool", "tool", "tool", "user"]
    asst = msgs[2]
    assert asst["content"] is None and [t["id"] for t in asst["tool_calls"]] == ["a1", "b2", "c3"] and all(t["type"] == "function" for t in asst["tool_calls"])
    assert [json.loads(t["function"]["arguments"]) for t in asst["tool_calls"]] == [{"channel": ch} for ch in ctx.channels[:3]]
    assert all(isinstance(t["function"]["arguments"], str) for t in asst["tool_calls"])  # arguments are a JSON STRING, as the format requires
    assert [m["tool_call_id"] for m in msgs[3:6]] == ["a1", "b2", "c3"]
    for m, ch in zip(msgs[3:6], ctx.channels):
        assert json.loads(m["content"])["channel"] == ch and json.loads(m["content"])["tool"] == "get_policy_decision"  # each result is the result of ITS call
    assert msgs[6] == {"role": "user", "content": A.WRAP_UP}  # the budget notice follows the tool messages as a user message


def test_calls_without_provider_ids_get_unique_ids_that_match_their_results():
    ctx = make_ctx()
    hist, calls, results, _ = conversation(ctx, ids=(None, None, None))
    c, _ = client([])
    msgs = c.messages("S", hist)
    ids = [t["id"] for t in msgs[2]["tool_calls"]]
    assert len(set(ids)) == 3 and [m["tool_call_id"] for m in msgs[3:6]] == ids


def test_duplicate_provider_ids_across_turns_are_made_unique():
    ctx = make_ctx()
    hist, calls, results, runner = conversation(ctx, n_calls=1, ids=("same",))
    again = ToolCall("get_quality_checks", {"channel": "meta"}, "same")
    hist += [{"role": "model", "turn": LLMTurn(tool_calls=[again])}, {"role": "tool", "results": [{"call": again, "payload": runner.execute(again)}]}]
    msgs = GroqClient(MODEL, CFG, transport=Fake([])).messages("S", hist)
    asst_ids = [t["id"] for m in msgs if m["role"] == "assistant" and m.get("tool_calls") for t in m["tool_calls"]]
    assert len(set(asst_ids)) == 2 and [m["tool_call_id"] for m in msgs if m["role"] == "tool"] == asst_ids


def test_a_tool_result_that_matches_no_call_or_misses_a_call_is_refused_before_sending():
    ctx = make_ctx()
    hist, calls, results, runner = conversation(ctx)
    c, _ = client([])
    stray = ToolCall("get_review_overview", {}, "zz")
    bad = copy.deepcopy(hist)
    bad[2] = {"role": "tool", "results": [{"call": stray, "payload": runner.execute(stray)}]}
    with pytest.raises(GroqError, match="matches no assistant tool call"):
        c.messages("S", bad)
    short = copy.deepcopy(hist)
    short[2]["results"] = short[2]["results"][:2]
    with pytest.raises(GroqError, match="do not match the assistant's tool calls"):
        c.messages("S", short)
    with pytest.raises(GroqError, match="follows tool calls that have no tool result"):
        c.messages("S", hist[:2] + [{"role": "user", "text": "oops"}])
    assert c.transport.bodies == []


# ---------------------------------------------------------------------------------------------- the model-visible compact view


def test_the_compact_view_changes_only_what_is_sent_never_the_authoritative_envelope():
    ctx = make_ctx()
    runner = T.ToolRunner(ctx, 40)
    env = runner.execute(ToolCall("get_economics", {"channel": "meta"}, "p"))
    before = copy.deepcopy(env)
    first, later = json.loads(model_view(env, True)), json.loads(model_view(env, False))
    assert env == before and runner.records["tc_001"] == before  # untouched, and still what the validator checks against
    for k in ("tool_call_id", "tool", "channel", "populations", "result"):
        assert first[k] == env[k] == later[k]
    assert "as_of" in first and "policy_hash" in first and "as_of" not in later and "policy_hash" not in later
    assert all("output_sha256" not in v and "args" not in v for v in (first, later))
    assert ", " not in model_view(env, False) and ": " not in model_view(env, False)  # compact separators
    assert len(model_view(env, False)) < len(json.dumps(env))
    err_env = runner.execute(ToolCall("get_economics", {"channel": "nope"}, "q"))
    assert json.loads(model_view(err_env, False))["error"]["code"] == "unknown_channel"  # errors are shown in full


def test_denied_calls_and_the_wrap_up_notice_reach_the_model():
    ctx = make_ctx()
    runner = T.ToolRunner(ctx, 2)
    calls = [ToolCall("get_policy_decision", {"channel": ch}, f"i{n}") for n, ch in enumerate(ctx.channels[:4])]
    results = [{"call": cl, "payload": runner.deny(cl) if runner.exhausted else runner.execute(cl)} for cl in calls]
    hist = [{"role": "user", "text": "s"}, {"role": "model", "turn": LLMTurn(tool_calls=calls)}, {"role": "tool", "results": results, "notice": A.WRAP_UP}]
    msgs = client([])[0].messages("S", hist)
    tools = [json.loads(m["content"]) for m in msgs if m["role"] == "tool"]
    assert [t.get("error", {}).get("code") for t in tools] == [None, None, "tool_budget_exhausted", "tool_budget_exhausted"] and msgs[-1]["content"] == A.WRAP_UP


# ---------------------------------------------------------------------------------------------- parsing responses


def test_a_parallel_tool_call_response_becomes_tool_calls_with_provider_ids_and_usage():
    msg = {"role": "assistant", "content": None, "reasoning": "hidden", "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "get_review_overview", "arguments": "{}"}},
        {"id": "call_2", "type": "function", "function": {"name": "get_policy_decision", "arguments": "{\"channel\": \"meta\"}"}}]}
    c, _ = client([resp(msg, usage=(1500, 90), headers={"x-ratelimit-remaining-tokens": "6000", "x-ratelimit-reset-tokens": "7.5s"})])
    turn = c.generate("S", plain_history(), T.declarations())
    assert [(t.name, t.args, t.provider_id) for t in turn.tool_calls] == [("get_review_overview", {}, "call_1"), ("get_policy_decision", {"channel": "meta"}, "call_2")]
    assert turn.text is None and turn.usage == {"prompt_tokens": 1500, "output_tokens": 90, "finish_reason": "stop"} and turn.raw is None
    log = c.request_log[0]
    assert log["usage"]["total_tokens"] == 1590 and log["headers"]["x-ratelimit-remaining-tokens"] == "6000" and log["observed_chars_per_token"] > 0


def test_arguments_that_are_not_valid_json_reach_the_tool_runner_as_a_visible_error_not_a_crash_or_a_silent_repair():
    for bad in ("{\"channel\": ", "[1,2]", "\"just a string\""):
        msg = {"content": None, "tool_calls": [{"id": "c", "type": "function", "function": {"name": "get_policy_decision", "arguments": bad}}]}
        turn = client([resp(msg)])[0].generate("S", plain_history(), T.declarations())
        env = T.ToolRunner(make_ctx(), 5).execute(turn.tool_calls[0])
        assert "_unparsed_arguments" in turn.tool_calls[0].args and env["error"]["code"] == "bad_arguments" and "result" not in env


@pytest.mark.parametrize("data", ["not json", "{}", "{\"choices\": []}", "{\"choices\": [{}]}", "{\"choices\": [{\"message\": \"text\"}]}", "{\"choices\": [{\"message\": {\"tool_calls\": \"x\"}}]}",
                                  "{\"choices\": [{\"message\": {\"tool_calls\": [{\"id\": \"a\", \"function\": {}}]}}]}", "{\"choices\": [{\"message\": {\"tool_calls\": [{\"id\": \"a\"}]}}]}"])
def test_malformed_responses_are_api_failures_with_a_label(data):
    with pytest.raises(GroqError) as e:
        client([HttpResult(200, {}, data)])[0].generate("S", plain_history(), T.declarations())
    assert e.value.kind == "malformed_response" and "MALFORMED RESPONSE" in str(e.value)


def test_empty_content_without_calls_is_an_empty_text_turn_for_the_agent_to_repair():
    turn = client([resp({"role": "assistant", "content": None})])[0].generate("S", plain_history(), T.declarations())
    assert turn.text == "" and turn.tool_calls == []
    cut = client([resp({"role": "assistant", "content": "{\"summary\": \"cut off"}, finish="length")])[0].generate("S", plain_history(), T.declarations())
    assert cut.usage["finish_reason"] == "length"  # truncation is visible in the record; the agent's malformed-output path handles the partial text


# ---------------------------------------------------------------------------------------------- pacing


def test_the_pacer_is_a_token_bucket_that_refills_continuously_and_never_waits_for_an_oversized_request():
    ck = Clock()
    p = TokenPacer(tpm=8000, rpm=30, margin_tokens=300, clock=ck.now, sleep=ck.sleep)
    assert p.cap == 7700 and p.rate == pytest.approx(8000 / 60)
    ev = p.acquire(5000)
    p.record(ev, 5200)  # the bucket holds 8000 - 5200 = 2800
    ck.t += 10  # refills 1333
    p.acquire(2000)  # 4133 available: no wait
    assert ck.slept == []
    ck.t += 5  # level 2133 + 667 = 2800 available; the next request needs 3000
    p.acquire(3000)
    assert len(ck.slept) == 1 and ck.slept[0] == pytest.approx(200 / (8000 / 60) + 0.05, abs=0.02)  # waits only for the 200 missing tokens (1.5 s), not a whole minute
    with pytest.raises(GroqError) as e:
        p.acquire(7701)
    assert e.value.kind == "oversized_preflight" and "Waiting cannot fix" in str(e.value) and len(ck.slept) == 1  # refused, and did not sleep


def test_the_pacer_honours_requests_per_minute_and_the_providers_remaining_tokens_header():
    ck = Clock()
    p = TokenPacer(tpm=8000, rpm=3, margin_tokens=300, clock=ck.now, sleep=ck.sleep)
    for _ in range(3):
        p.record(p.acquire(100), 100)
        ck.t += 1
    p.acquire(100)
    assert ck.slept and ck.slept[0] == pytest.approx(60 - 3 + 0.05, abs=0.2)
    ck2 = Clock()
    p2 = TokenPacer(tpm=8000, rpm=30, margin_tokens=300, clock=ck2.now, sleep=ck2.sleep)
    p2.record(p2.acquire(100), 100, {"x-ratelimit-remaining-tokens": "500", "x-ratelimit-reset-tokens": "56.25s", "x-ratelimit-remaining-requests": "990"})
    assert p2.server["remaining_requests_day"] == 990 and p2.level == 500  # the provider's figure REPLACES our own level
    ck2.t += 1
    p2.acquire(2000)  # 500 + 133 available, 2000 needed: wait for the missing 1367 tokens
    assert ck2.slept[0] == pytest.approx(1367 / (8000 / 60) + 0.05, abs=0.05)


def test_a_rejected_request_gives_back_its_reservation_unless_the_provider_reports_a_bucket():
    ck = Clock()
    p = TokenPacer(clock=ck.now, sleep=ck.sleep)
    p.rejected(p.acquire(6000))  # no headers: it used nothing
    p.acquire(6000)
    assert ck.slept == []
    ck2 = Clock()
    p2 = TokenPacer(clock=ck2.now, sleep=ck2.sleep)
    p2.rejected(p2.acquire(3000), {"x-ratelimit-remaining-tokens": "1000", "x-ratelimit-reset-tokens": "52.5s"})  # a failed generation that still consumed tokens
    p2.acquire(2000)
    assert ck2.slept[0] == pytest.approx(1000 / (8000 / 60) + 0.05, abs=0.05)


def test_the_bucket_model_reproduces_every_rate_limit_header_the_live_service_returned():
    """Data test: in every recorded live response, reset-tokens x (limit/60) equals the tokens the bucket had used (limit - remaining-tokens): a continuously refilling bucket."""
    seen = 0
    for rel in ("reports/groq_smoke_status.json", "reports/phase5_runs/live_groq_2025-11-10.json"):
        f = ROOT / rel
        if not f.exists():
            continue
        for r in json.loads(f.read_text(encoding="utf-8")).get("request_log", []):
            h = r.get("headers", {})
            if "x-ratelimit-remaining-tokens" in h:
                used = 8000 - int(h["x-ratelimit-remaining-tokens"])
                assert parse_duration(h["x-ratelimit-reset-tokens"]) * 8000 / 60 == pytest.approx(used, abs=3)
                seen += 1
    if not seen:
        pytest.skip("no live responses recorded in this checkout")


# ---------------------------------------------------------------------------------------------- generate(): retries, limits, timeouts, headroom


def ok_msg():
    return resp({"role": "assistant", "content": "ok"})


def test_a_per_minute_429_waits_the_providers_retry_after_then_succeeds():
    c, ck = client([err(429, "Rate limit reached on tokens per minute (TPM). Please try again in 3s.", {"retry-after": "3"}), ok_msg()])
    turn = c.generate("S", plain_history(), T.declarations())
    assert turn.text == "ok" and len(c.transport.bodies) == 2 and 3.4 < ck.slept[0] < 3.6 and c.request_log[0]["attempts"] == 2


def test_retries_are_bounded_and_a_long_wait_is_not_waited_for():
    c, ck = client([err(429, "tokens per minute. try again in 1s", {"retry-after": "1"})] * 5)
    with pytest.raises(GroqError) as e:
        c.generate("S", plain_history(), T.declarations())
    assert e.value.kind == "per_minute" and len(c.transport.bodies) == 3 and "after 2 bounded retries" in str(e.value)  # 1 try + max_rate_limit_retries
    c2, ck2 = client([err(429, "tokens per minute", {"retry-after": "500"}), ok_msg()])
    with pytest.raises(GroqError):
        c2.generate("S", plain_history(), T.declarations())
    assert len(c2.transport.bodies) == 1 and ck2.slept == []


@pytest.mark.parametrize("status,msg,kind,label", [
    (429, "Rate limit reached on tokens per day (TPD): Limit 200000. Please try again in 9m30s.", "daily", "DAILY LIMIT REACHED"),
    (429, "Rate limit reached on requests per day (RPD): Limit 1000.", "daily", "DAILY LIMIT REACHED"),
    (413, "Request too large for model on tokens per minute (TPM): Limit 8000, Requested 9500", "request_too_large", "REQUEST TOO LARGE (HTTP 413)"),
    (401, "Invalid API Key", "auth", "AUTHENTICATION FAILED"), (402, "payment required", "billing", "BILLING"), (400, "invalid request body", "bad_request", "REQUEST REJECTED"),
])
def test_daily_limits_oversize_auth_and_billing_errors_are_never_retried(status, msg, kind, label):
    c, ck = client([err(status, msg), ok_msg()])
    with pytest.raises(GroqError) as e:
        c.generate("S", plain_history(), T.declarations())
    assert e.value.kind == kind and label in str(e.value) and len(c.transport.bodies) == 1 and ck.slept == []


def test_server_errors_get_a_short_bounded_retry():
    c, ck = client([err(503, "over capacity"), ok_msg()])
    assert c.generate("S", plain_history(), T.declarations()).text == "ok" and len(ck.slept) == 1 and ck.slept[0] < 10
    c2, _ = client([err(500, "boom")] * 4)
    with pytest.raises(GroqError) as e:
        c2.generate("S", plain_history(), T.declarations())
    assert e.value.kind == "server" and len(c2.transport.bodies) == 3


def test_timeouts_are_labelled_not_retried_and_use_seconds():
    c, ck = client([TimeoutError("read timed out"), ok_msg()], cfg={"timeout_seconds": 7})
    with pytest.raises(GroqError) as e:
        c.generate("S", plain_history(), T.declarations())
    assert e.value.kind == "timeout" and "TIMEOUT after the configured 7s" in str(e.value) and len(c.transport.bodies) == 1 and c.transport.timeouts == [7.0]  # seconds, not 7000
    c2, _ = client([ConnectionResetError("reset"), ok_msg()])
    with pytest.raises(GroqError) as e2:
        c2.generate("S", plain_history(), T.declarations())
    assert e2.value.kind == "network" and len(c2.transport.bodies) == 1
    with pytest.raises(ValueError):
        client([], cfg={"timeout_seconds": 0})


def test_output_headroom_shrinks_the_completion_cap_and_an_oversized_prompt_is_refused_unsent():
    ctx = make_ctx()
    c, _ = client([ok_msg()])
    hist = plain_history("x" * 13000)  # about 3250 tokens at 4.0 chars/token on top of the system prompt and tool schemas (about 2100)
    c.generate(A.SYSTEM_PROMPT, hist, T.declarations())
    sent = c.transport.bodies[0]["max_completion_tokens"]
    est = c.request_log[0]["prompt_tokens_estimated"]
    assert sent == c.request_log[0]["max_completion_tokens"] and est + sent <= c.pacer.cap and CFG["min_completion_tokens"] <= sent < CFG["max_completion_tokens"]
    big, _ = client([ok_msg()])
    with pytest.raises(GroqError) as e:
        big.generate(A.SYSTEM_PROMPT, plain_history("x" * 30000), T.declarations())
    assert e.value.kind == "oversized_preflight" and "NOT sent" in str(e.value) and big.transport.bodies == []


def test_the_token_estimate_is_calibrated_from_the_first_response():
    c, _ = client([resp({"content": "ok"}, usage=(2000, 10)), ok_msg()])
    c.generate("S", plain_history(), T.declarations())
    first_ratio = c.request_log[0]["observed_chars_per_token"]
    assert c.chars_per_token == pytest.approx(first_ratio * 0.95, rel=0.01) and c.chars_per_token != CFG["assumed_chars_per_token"]


# ---------------------------------------------------------------------------------------------- key handling, mode and provider


def test_the_key_is_read_only_from_the_environment_and_never_appears_in_errors_records_or_reprs(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(GroqError, match="GROQ_API_KEY is not set"):
        GroqClient(MODEL, CFG)
    secret = "gsk_TESTKEY_not_real_0123456789abcdef"
    monkeypatch.setenv("GROQ_API_KEY", secret)
    live = GroqClient(MODEL, CFG)
    assert live.mode == "live" and live.provider == "groq" and secret not in repr(live.transport) and secret not in repr(vars(live)).replace(repr(live._key), "")
    fake_c = GroqClient(MODEL, CFG, transport=Fake([err(401, f"Invalid API Key {secret}")]))
    assert fake_c.mode == "mock"  # an injected transport can never be reported as live
    with pytest.raises(GroqError) as e:
        fake_c.generate("S", plain_history(), T.declarations())
    assert secret not in str(e.value) and "<key>" in str(e.value)
    assert secret not in json.dumps(fake_c.request_log)


def test_the_only_http_code_path_is_the_transport_and_it_sends_the_key_only_in_the_authorization_header(monkeypatch):
    seen = {}

    class R:
        status, headers = 200, {"X-Ratelimit-Remaining-Tokens": "7000"}

        def read(self):
            return b"{}"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout):
        seen.update(url=req.full_url, headers={k.lower(): v for k, v in req.header_items()}, timeout=timeout, body=json.loads(req.data))
        return R()

    monkeypatch.setattr(G.urllib.request, "urlopen", fake_urlopen)
    t = G.UrllibTransport("gsk_fake_for_test")
    res = t({"model": MODEL, "messages": []}, 12.0)
    assert seen["url"] == G.ENDPOINT and seen["timeout"] == 12.0 and seen["headers"]["authorization"] == "Bearer gsk_fake_for_test" and "gsk_fake_for_test" not in json.dumps(seen["body"])
    assert res.headers == {"x-ratelimit-remaining-tokens": "7000"} and seen["headers"]["user-agent"] == G.USER_AGENT


# ---------------------------------------------------------------------------------------------- end to end through the agent loop (faked transport)


BIG = {"tpm": 100000}  # plumbing tests are about the loop, not the 8000-token pacing (that is tested above)


def run_with(items, ctx=None, **kw):
    ctx = ctx or make_ctx()
    c, ck = client(items, cfg=BIG)
    return ctx, c, A.run_agent(ctx, c, max_tool_calls=kw.pop("max_tool_calls", 14), **kw)


def test_a_valid_answer_through_the_adapter_is_an_agent_proposal_labelled_mock_provider_groq():
    ctx = make_ctx()
    s, turns = standard_conversation(ctx)
    _, c, run = run_with([from_turn(t) for t in turns], ctx)
    assert run.status == "agent_proposal" and run.validation["ok"] and run.repairs_used == 0 and run.rejections == []
    assert run.mode == "mock" and run.provider == "groq" and run.model == "openai/gpt-oss-120b"  # faked transport: never 'live'
    assert any(l.startswith("MODE: MOCK (provider groq, model openai/gpt-oss-120b)") and "NOT evidence" in l for l in run.labels)
    d = run.to_dict()
    assert d["provider"] == "groq" and d["mode"] == "mock" and d["usage"]["prompt_tokens"] > 0
    # the calls the model made were answered with matching ids in the second request
    second = c.transport.bodies[1]["messages"]
    asst = next(m for m in second if m["role"] == "assistant")
    assert [m["tool_call_id"] for m in second if m["role"] == "tool"] == [t["id"] for t in asst["tool_calls"]]
    assert run.tool_calls and all(t["status"] == "ok" for t in run.tool_calls)


def test_the_validator_still_rejects_a_fabricated_number_then_one_repair_then_the_rules_only_fallback():
    ctx = make_ctx()

    def mutate(items, s):
        items[1]["claims"][0]["value"] = "increase_test"  # google_uac's policy action is decrease

    s, turns = standard_conversation(ctx, mutate)
    answer = turns[1]
    _, c, run = run_with([from_turn(turns[0]), from_turn(answer), from_turn(answer)], ctx)
    assert run.status == "rules_only_fallback" and run.fallback_reason == "validation_failed" and run.repairs_used == 1
    assert any("RULES-ONLY FALLBACK: AGENT UNAVAILABLE (AGENT OUTPUT REJECTED BY THE VALIDATOR)" in l for l in run.labels) and run.fallback_validation["ok"]
    assert run.provider == "groq" and run.mode == "mock"
    assert [r["model_turn"] for r in run.rejections] == [2, 3] and all(r["errors"] and r["n_errors"] >= 1 for r in run.rejections)  # BOTH rejected answers are on record, not just the last
    assert run.to_dict()["rejections"][0]["errors"][0]["code"] and json.dumps(run.to_dict())
    repair_request = c.transport.bodies[2]["messages"]
    assert repair_request[-1]["role"] == "user" and "REJECTED by the validator" in repair_request[-1]["content"]


def test_a_reply_that_is_not_json_twice_takes_the_malformed_output_fallback():
    ctx = make_ctx()
    s, turns = standard_conversation(ctx)
    _, _, run = run_with([from_turn(turns[0]), from_turn(text_turn("I think we should raise everything")), from_turn(text_turn("still not json"))], ctx)
    assert run.status == "rules_only_fallback" and run.fallback_reason == "malformed_output"


@pytest.mark.parametrize("failure,reason_text", [
    (err(429, "Rate limit reached on tokens per day (TPD): Limit 200000"), "DAILY LIMIT REACHED"),
    (err(413, "Request too large on tokens per minute (TPM): Limit 8000, Requested 9000"), "REQUEST TOO LARGE (HTTP 413)"),
    (TimeoutError("slow"), "TIMEOUT after the configured"),
    (HttpResult(200, {}, "garbage"), "MALFORMED RESPONSE"),
])
def test_a_provider_failure_takes_the_labelled_api_failure_fallback_and_still_passes_the_validator(failure, reason_text):
    ctx, _, run = run_with([failure])
    assert run.status == "rules_only_fallback" and run.fallback_reason == "api_failure" and reason_text in str(run.fallback_detail)
    assert run.fallback_validation["ok"] and any("AGENT UNAVAILABLE (API FAILURE)" in l for l in run.labels) and run.provider == "groq"


def test_a_preflight_refusal_is_an_api_failure_fallback_not_a_crash_and_nothing_is_sent():
    ctx = make_ctx()
    c = GroqClient(MODEL, {**CFG, "tpm": 2000}, transport=Fake([ok_msg()]))  # the system prompt and tool schemas alone exceed this minute's tokens
    run = A.run_agent(ctx, c, max_tool_calls=14)
    assert run.status == "rules_only_fallback" and run.fallback_reason == "api_failure" and "REQUEST TOO LARGE (preflight)" in str(run.fallback_detail) and c.transport.bodies == []


def test_the_tool_budget_wrap_up_notice_is_sent_after_the_tool_messages():
    ctx = make_ctx()
    s, turns = standard_conversation(ctx)
    _, c, run = run_with([from_turn(turns[0]), from_turn(text_turn("{}")), from_turn(text_turn("{}"))], ctx, max_tool_calls=3)
    second = c.transport.bodies[1]["messages"]
    assert second[-1] == {"role": "user", "content": A.WRAP_UP} and [m["role"] for m in second[-8:-1]].count("tool") >= 7
    denied = [json.loads(m["content"]) for m in second if m["role"] == "tool"][3:]
    assert denied and all(d["error"]["code"] == "tool_budget_exhausted" for d in denied) and any("TOOL BUDGET REACHED" in l for l in run.labels)


# ---------------------------------------------------------------------------------------------- labels: provider, model, mode stay distinct


def test_gemini_and_mock_labels_and_records_are_unchanged_and_providers_are_distinct():
    ctx = make_ctx()
    s, llm = standard_conversation(ctx)[0], MockLLM(standard_conversation(ctx)[1])
    run = A.run_agent(ctx, llm)
    assert run.provider == "mock" and run.labels[0] == "MODE: MOCK (mock-llm (scripted)). Scripted plumbing test: NOT evidence about any model or agent."
    assert A.mode_label("live", "models/x") == "MODE: LIVE (model models/x)." and A.mode_label("live", "models/x", "gemini") == "MODE: LIVE (model models/x)."
    assert A.mode_label("live", "m", "groq") == "MODE: LIVE (provider groq, model m)." and "REPLAY" in A.mode_label("replay", "m", "groq") and "MOCK" in A.mode_label("mock", "m", "groq")
    from src.agent.llm import GeminiClient, ReplayLLM
    assert GeminiClient.provider == "gemini" and MockLLM([]).provider == "mock" and GroqClient.provider == "groq"
    assert ReplayLLM.from_turns([]).provider is None


def test_a_recording_remembers_its_provider_and_only_a_live_run_can_be_recorded():
    live_run = {"mode": "live", "provider": "groq", "model": "openai/gpt-oss-120b", "model_turns": [{"text": "x", "tool_calls": []}], "as_of": "2025-10-13", "policy_hash": "h",
                "tool_calls": [], "status": "agent_proposal"}
    assert recording_from_run(live_run)["provider"] == "groq"
    with pytest.raises(ValueError):
        recording_from_run({**live_run, "mode": "mock"})  # a faked-transport Groq run is mode 'mock' and can never become a recording


# ---------------------------------------------------------------------------------------------- the model's own tool-call format failure (seen live)

OBSERVED = "Tool call validation failed: tool call validation failed: attempted to call tool 'commentary' which was not in request.tools"
BUCKET = {"x-ratelimit-remaining-tokens": "4383", "x-ratelimit-reset-tokens": "27.127s", "x-ratelimit-remaining-requests": "994"}


def test_the_observed_live_failure_is_classified_as_a_model_tool_call_format_failure():
    for body in (err(400, OBSERVED), HttpResult(400, {}, json.dumps({"error": {"message": "x", "type": "invalid_request_error", "code": "tool_use_failed", "failed_generation": "<|channel|>commentary"}}))):
        assert classify_error(body).kind == "tool_use_failed"
    assert classify_error(err(400, "invalid request body")).kind == "bad_request"


def test_tool_use_failed_is_retried_a_bounded_number_of_times_and_a_later_success_is_used():
    c, ck = client([err(400, OBSERVED, BUCKET), ok_msg()], cfg={"max_tool_use_failed_retries": 2})
    turn = c.generate("S", plain_history(), T.declarations())
    assert turn.text == "ok" and len(c.transport.bodies) == 2 and c.request_log[0]["attempts"] == 2 and c.request_log[0]["error_kind"] == "tool_use_failed"
    assert "commentary" in c.request_log[0]["error_excerpt"] and ck.slept and ck.slept[0] < 5


def test_tool_use_failed_that_persists_takes_the_labelled_fallback_after_the_bound_and_keeps_the_provider_message():
    c, _ = client([err(400, OBSERVED, BUCKET)] * 5, cfg={"max_tool_use_failed_retries": 2})
    with pytest.raises(GroqError) as e:
        c.generate("S", plain_history(), T.declarations())
    assert e.value.kind == "tool_use_failed" and len(c.transport.bodies) == 3 and "after 2 bounded retries" in str(e.value) and "commentary" in str(e.value)  # 1 try + max_tool_use_failed_retries
    c0, _ = client([err(400, OBSERVED, BUCKET)] * 2, cfg={"max_tool_use_failed_retries": 0})
    with pytest.raises(GroqError):
        c0.generate("S", plain_history(), T.declarations())
    assert len(c0.transport.bodies) == 1


def test_a_failed_generation_that_consumed_tokens_is_charged_to_the_local_bucket_from_the_headers():
    c, ck = client([err(400, OBSERVED, BUCKET)] * 3, cfg={"max_tool_use_failed_retries": 0})
    with pytest.raises(GroqError):
        c.generate("S", plain_history(), T.declarations())
    assert c.pacer.server["remaining_tokens"] == 4383 and c.pacer.level == 4383  # the provider says 3,617 tokens went, although no usage figure came back


def test_a_conversation_survives_one_tool_call_format_failure_end_to_end():
    ctx = make_ctx()
    s, turns = standard_conversation(ctx)
    items = [from_turn(turns[0]), err(400, OBSERVED, BUCKET), from_turn(turns[1])]
    c, _ = client(items, cfg={**BIG, "max_tool_use_failed_retries": 2})
    run = A.run_agent(ctx, c, max_tool_calls=14)
    assert run.status == "agent_proposal" and run.provider == "groq" and run.mode == "mock"
    assert [r.get("error_kind") for r in c.request_log] == [None, "tool_use_failed"] and c.request_log[1]["attempts"] == 2


def test_the_bucket_never_holds_more_than_one_minute_of_tokens_however_long_it_idled():
    ck = Clock()
    p = TokenPacer(tpm=8000, rpm=30, margin_tokens=300, clock=ck.now, sleep=ck.sleep)
    ck.t += 100000  # idle for a long time
    assert p.available() == 8000
    p.acquire(7700)
    assert ck.slept == []
    p.acquire(7700)  # the bucket is nearly empty again: this must wait about 57 s, not run free on 'saved up' tokens
    assert ck.slept and ck.slept[0] == pytest.approx((7700 - 300) / (8000 / 60), abs=0.5)


def test_every_request_record_names_the_provider_note_it_used():
    c, _ = client([ok_msg()])
    c.generate("S", plain_history(), T.declarations())
    import hashlib
    assert c.request_log[0]["provider_note_sha256"] == hashlib.sha256(G.PROVIDER_NOTE.encode()).hexdigest()[:16]


def test_the_shipped_turn_bound_covers_a_single_call_per_turn_investigation_plus_a_repair():
    assert CFG["max_model_turns"] >= CFG["max_tool_calls"] + 2 + 1 - 1  # every call in its own turn, the answer, and one repair


# ---------------------------------------------------------------------------------------------- a final answer the provider mis-parsed as a call to a non-existent tool (seen live: 'JSON')

ANSWER = {"summary": "One decrease, holds elsewhere.", "summary_claims": [], "items": [{"channel": "meta", "action": "hold", "new_budget": 90.0}]}
NAMES = [t.name for t in T.TOOLS]


def wrapped_error(name="JSON", arguments=None, as_string=False):
    args = ANSWER if arguments is None else arguments
    fg = json.dumps({"name": name, "arguments": json.dumps(args) if as_string else args}, indent=1)
    return HttpResult(400, BUCKET, json.dumps({"error": {"message": f"Tool call validation failed: attempted to call tool '{name}' which was not in request.tools", "type": "invalid_request_error",
                                                       "code": "tool_use_failed", "failed_generation": fg}}))


def test_the_shipped_config_does_not_retry_an_identical_failing_request_and_unwraps_a_wrapped_final_answer():
    assert CFG["max_tool_use_failed_retries"] == 0 and CFG["unwrap_wrapped_final_answer"] is True


def test_unwrap_returns_the_arguments_unchanged_only_for_a_non_tool_name_with_the_answers_two_keys():
    assert json.loads(G.unwrap_failed_generation(wrapped_error().text, NAMES)) == ANSWER
    assert json.loads(G.unwrap_failed_generation(wrapped_error(as_string=True).text, NAMES)) == ANSWER  # arguments given as a JSON string
    assert G.unwrap_failed_generation(wrapped_error(name="get_policy_decision").text, NAMES) is None  # a real tool whose call failed validation is NOT an answer
    assert G.unwrap_failed_generation(wrapped_error(arguments={"summary": "x"}).text, NAMES) is None  # not the answer's shape: nothing is completed or guessed
    assert G.unwrap_failed_generation(wrapped_error(arguments={"channel": "meta"}).text, NAMES) is None
    assert G.unwrap_failed_generation(err(400, OBSERVED).text, NAMES) is None  # the first live failure ('commentary') carried no usable answer
    cut = json.loads(wrapped_error().text)
    cut["error"]["failed_generation"] = cut["error"]["failed_generation"][:40]  # truncated text is not repaired
    assert G.unwrap_failed_generation(json.dumps(cut), NAMES) is None
    assert G.unwrap_failed_generation("not json", NAMES) is None


def test_a_wrapped_final_answer_becomes_a_text_turn_without_a_retry_and_is_counted_and_labelled():
    c, ck = client([wrapped_error(), ok_msg()])
    turn = c.generate("S", plain_history(), T.declarations())
    assert turn.tool_calls == [] and json.loads(turn.text) == ANSWER and turn.usage["finish_reason"] == "unwrapped_from_tool_use_failed"
    assert len(c.transport.bodies) == 1 and ck.slept == [] and c.salvaged == 1 and c.request_log[0]["unwrapped_failed_generation"] is True and "error_excerpt" in c.request_log[0]
    assert c.pacer.level == 4383  # the failed generation's tokens are charged from the provider's headers
    assert c.run_notes() and "PROVIDER FORMAT RECOVERY: 1 model answer" in c.run_notes()[0]
    off, _ = client([wrapped_error()], cfg={"unwrap_wrapped_final_answer": False})
    with pytest.raises(GroqError) as e:
        off.generate("S", plain_history(), T.declarations())
    assert e.value.kind == "tool_use_failed" and off.salvaged == 0 and off.run_notes() == []


def test_an_unwrapped_answer_still_goes_through_the_validator_and_the_run_says_so():
    ctx = make_ctx()
    s, turns = standard_conversation(ctx)
    good = json.loads(turns[1].text)
    _, c, run = run_with([from_turn(turns[0]), wrapped_error(arguments=good)], ctx)
    assert run.status == "agent_proposal" and run.validation["ok"] and any(l.startswith("PROVIDER FORMAT RECOVERY") for l in run.labels)
    # a wrapped answer that is WRONG is rejected like any other, repaired once, and the run still carries the label
    bad = json.loads(turns[1].text)
    bad["items"][1]["claims"][0]["value"] = "increase_test"
    _, c2, run2 = run_with([from_turn(turns[0]), wrapped_error(arguments=bad), from_turn(turns[1])], ctx)
    assert run2.status == "agent_proposal" and run2.repairs_used == 1 and any(l.startswith("PROVIDER FORMAT RECOVERY") for l in run2.labels)
    _, c3, run3 = run_with([from_turn(turns[0]), wrapped_error(arguments=bad), wrapped_error(arguments=bad)], ctx)
    assert run3.status == "rules_only_fallback" and run3.fallback_reason == "validation_failed" and any(l.startswith("PROVIDER FORMAT RECOVERY") for l in run3.labels)


def test_mock_and_gemini_style_runs_carry_no_recovery_label():
    ctx = make_ctx()
    run = A.run_agent(ctx, MockLLM(standard_conversation(ctx)[1]))
    assert not any("PROVIDER FORMAT RECOVERY" in l for l in run.labels)
    plain_ctx = make_ctx()
    _, _, run2 = run_with([from_turn(t) for t in standard_conversation(plain_ctx)[1]], plain_ctx)
    assert run2.status == "agent_proposal" and not any("PROVIDER FORMAT RECOVERY" in l for l in run2.labels)  # the label only appears when a recovery happened


# ---------------------------------------------------------------------------------------------- the live runner keeps earlier attempts


def test_the_groq_live_runner_keeps_every_earlier_attempt_in_its_status_history(tmp_path, monkeypatch):
    from scripts import run_agent as RA
    monkeypatch.setenv("GROQ_API_KEY", "gsk_fake_runner_test")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "agent_groq.yaml").write_text((ROOT / "config" / "agent_groq.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(RA, "ROOT", tmp_path), monkeypatch.setattr(RA, "RUNS", tmp_path / "runs"), monkeypatch.setattr(RA, "DEMO", tmp_path / "demo")
    monkeypatch.setattr(RA, "GROQ_STATUS", tmp_path / "groq_live_status.json")
    monkeypatch.setattr(RA, "build_environment", lambda as_of: make_ctx())
    monkeypatch.setattr(RA, "GroqClient", lambda model, cfg: GroqClient(model, {**cfg, "tpm": 100000}, transport=Fake([err(401, "Invalid API Key")] * 3)))
    for _ in range(3):
        assert RA.mode_live_groq(["2025-11-10"]) == 0
    st = json.loads((tmp_path / "groq_live_status.json").read_text(encoding="utf-8"))
    assert st["attempt"] == 3 and len(st["history"]) == 2 and all(h["live_agent_runs"][0]["status"] == "rules_only_fallback" for h in st["history"])
    assert sorted(f.name for f in (tmp_path / "runs").glob("live_groq_*")) == ["live_groq_2025-11-10.json", "live_groq_2025-11-10_2.json", "live_groq_2025-11-10_3.json"]  # no run file is overwritten
    assert "gsk_fake_runner_test" not in (tmp_path / "groq_live_status.json").read_text(encoding="utf-8")


def test_a_malformed_answer_is_recorded_as_a_rejection_too():
    ctx = make_ctx()
    s, turns = standard_conversation(ctx)
    _, _, run = run_with([from_turn(turns[0]), from_turn(text_turn("{\"summary\": \"x\", \"items\": [{\"new_budget\": \"   \n\n\"}]")), from_turn(text_turn("still broken"))], ctx)
    assert run.fallback_reason == "malformed_output" and [r["errors"][0]["code"] for r in run.rejections] == ["malformed_output", "malformed_output"]
