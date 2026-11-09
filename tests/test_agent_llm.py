"""GeminiClient with a FAKE transport. These tests exercise the production conversion, config, error and redaction code. They say nothing about live
Gemini: a client built around an injected fake reports mode 'mock', never 'live'. No credentials or network are used."""
import json

import pytest
from google.genai import types

from src.agent import agent as A
from src.agent import tools as T
from src.agent.llm import GeminiClient, LLMError, LLMTurn, ToolCall, candidate_models, list_generate_models, recording_from_run
from tests.agent_helpers import make_ctx, standard_conversation


def text_response(text):
    return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part(text=text)]))])


def call_response(calls):
    parts = [types.Part(function_call=types.FunctionCall(name=n, args=a, id=i)) for n, a, i in calls]
    return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model", parts=parts))])


class FakeClient:
    def __init__(self, script):
        self.script, self.calls, self.models = list(script), [], self

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item(kwargs["contents"]) if callable(item) else item

    def list(self):
        return [types.Model(name="models/gemini-x-flash", supported_actions=["generateContent"]), types.Model(name="models/embed-1", supported_actions=["embedContent"])]


def decls():
    return T.declarations()


def test_a_client_around_a_fake_transport_is_never_reported_as_live():
    assert GeminiClient("some-model", client=FakeClient([])).mode == "mock"
    with pytest.raises(ValueError, match="only an actual LIVE run"):
        recording_from_run({"mode": GeminiClient("m", client=FakeClient([])).mode, "model": "m", "model_turns": [{}]})


def test_requests_carry_the_system_prompt_all_seven_tool_declarations_and_disable_automatic_function_calling():
    fake = FakeClient([text_response("{}")])
    GeminiClient("m", client=fake).generate("SYSTEM TEXT", [{"role": "user", "text": "hello"}], decls())
    req = fake.calls[0]
    cfg = req["config"]
    assert req["model"] == "m" and cfg.system_instruction == "SYSTEM TEXT" and cfg.temperature == 0.0
    assert cfg.automatic_function_calling.disable is True  # the harness, not the SDK, executes tools
    names = [d.name for t in cfg.tools for d in t.function_declarations]
    assert sorted(names) == sorted(T.REGISTRY)
    plan = next(d for t in cfg.tools for d in t.function_declarations if d.name == "plan_experiment")
    assert set(plan.parameters.properties) == {"channel", "new_budget"} and plan.parameters.required == ["channel", "new_budget"]
    constraints = next(d for t in cfg.tools for d in t.function_declarations if d.name == "check_budget_constraints")
    assert constraints.parameters.properties["changes"].type == types.Type.ARRAY and constraints.parameters.properties["changes"].items.type == types.Type.OBJECT


def test_tool_calls_are_returned_and_results_are_sent_back_as_function_responses_with_the_models_own_content_preserved():
    fake = FakeClient([call_response([("get_policy_decision", {"channel": "meta"}, "g-1")]), text_response('{"summary": "x", "items": []}')])
    client = GeminiClient("m", client=fake)
    turn = client.generate("S", [{"role": "user", "text": "go"}], decls())
    assert turn.text is None and turn.tool_calls == [ToolCall("get_policy_decision", {"channel": "meta"}, "g-1")] and turn.raw is not None
    history = [{"role": "user", "text": "go"}, {"role": "model", "turn": turn},
               {"role": "tool", "results": [{"call": turn.tool_calls[0], "payload": {"tool_call_id": "tc_001", "result": {"action": "hold"}}}]}]
    final = client.generate("S", history, decls())
    sent = fake.calls[1]["contents"]
    assert [c.role for c in sent] == ["user", "model", "user"]
    assert sent[1] is turn.raw  # the provider's own content goes back unchanged (call ids, thought signatures)
    fr = sent[2].parts[0].function_response
    assert fr.name == "get_policy_decision" and fr.id == "g-1" and fr.response["tool_call_id"] == "tc_001" and fr.response["result"] == {"action": "hold"}
    assert final.text == '{"summary": "x", "items": []}' and final.tool_calls == []


def test_every_provider_failure_becomes_llmerror_and_the_key_is_redacted(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "sk-test-SECRET-123")
    client = GeminiClient("m", client=FakeClient([RuntimeError("401 bad key sk-test-SECRET-123 rejected")]))
    with pytest.raises(LLMError) as exc:
        client.generate("S", [{"role": "user", "text": "x"}], decls())
    assert "sk-test-SECRET-123" not in str(exc.value) and "<key>" in str(exc.value) and "RuntimeError" in str(exc.value)


def test_without_a_key_and_without_an_injected_client_nothing_is_attempted(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="GEMINI_API_KEY is not set"):
        GeminiClient("m")


def test_an_empty_or_blocked_reply_is_returned_as_an_empty_turn_not_an_exception():
    fake = FakeClient([types.GenerateContentResponse(candidates=[])])
    turn = GeminiClient("m", client=fake).generate("S", [{"role": "user", "text": "x"}], decls())
    assert turn.text is None and turn.tool_calls == [] and turn.raw is None


def test_a_full_agent_run_over_a_fake_gemini_transport_is_still_mock_and_the_api_error_path_falls_back():
    ctx = make_ctx()
    s, turns = standard_conversation(ctx)

    def respond(item):
        if item.tool_calls:
            return call_response([(c.name, c.args, f"g-{i}") for i, c in enumerate(item.tool_calls)])
        return text_response(item.text)

    fake = FakeClient([respond(turns[0]), respond(turns[1])])
    run = A.run_agent(ctx, GeminiClient("fake-model", client=fake))
    assert run.mode == "mock" and run.status == "agent_proposal" and run.proposal.source == "agent"
    down = A.run_agent(make_ctx(), GeminiClient("fake-model", client=FakeClient([ConnectionError("network unreachable")])))
    assert down.status == "rules_only_fallback" and down.fallback_reason == "api_failure" and "network unreachable" in down.fallback_detail


# ------------------------------------------------------------------------------------------------ timeout


def test_the_configured_timeout_in_seconds_is_converted_to_the_sdk_unit_milliseconds_and_validated():
    from src.agent.llm import http_timeout_ms
    assert (http_timeout_ms(60), http_timeout_ms(45), http_timeout_ms(1.5), http_timeout_ms(0.25)) == (60000, 45000, 1500, 250)
    for bad in (0, -1, float("inf"), float("nan"), True, "60", None):
        with pytest.raises(ValueError, match="timeout_seconds"):
            http_timeout_ms(bad)
    assert types.HttpOptions.model_fields["timeout"].description.endswith("milliseconds.")  # the installed SDK's documented unit


def test_the_timeout_is_supplied_on_every_request():
    fake = FakeClient([text_response("{}"), text_response("{}")])
    GeminiClient("m", client=fake, timeout_seconds=45).generate("S", [{"role": "user", "text": "x"}], decls())
    GeminiClient("m", client=fake).generate("S", [{"role": "user", "text": "x"}], decls())  # the default is the configuration's 60 s
    assert fake.calls[0]["config"].http_options.timeout == 45000 and fake.calls[1]["config"].http_options.timeout == 60000


def test_the_live_client_is_constructed_with_the_timeout_without_touching_the_network(monkeypatch):
    seen = {}

    class StubSdkClient:  # replaces genai.Client so nothing can reach the network
        def __init__(self, **kwargs):
            seen.update(kwargs)
            self.models = FakeClient([text_response("{}")])

    import google.genai as genai
    monkeypatch.setattr(genai, "Client", StubSdkClient)
    monkeypatch.setenv("GEMINI_API_KEY", "sk-test-not-a-real-key")
    client = GeminiClient("m", timeout_seconds=30)
    assert client.mode == "live" and seen["api_key"] == "sk-test-not-a-real-key" and seen["http_options"].timeout == 30000
    client.generate("S", [{"role": "user", "text": "x"}], decls())
    assert client.client.models.calls[0]["config"].http_options.timeout == 30000  # client-level AND request-level


def test_the_agent_configuration_timeout_reaches_the_live_path():
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "agent.yaml").read_text(encoding="utf-8"))
    from src.agent.llm import http_timeout_ms
    assert cfg["timeout_seconds"] == 60 and http_timeout_ms(cfg["timeout_seconds"]) == 60000
    src = (Path(__file__).resolve().parents[1] / "scripts" / "run_agent.py").read_text(encoding="utf-8")
    assert 'http_timeout_ms(cfg["timeout_seconds"])' in src and 'timeout_seconds=cfg["timeout_seconds"]' in src  # both the model-selection client and the agent client


def test_a_timeout_takes_the_existing_labelled_api_failure_fallback():
    import requests
    for exc in (TimeoutError("timed out"), requests.exceptions.ReadTimeout("read timed out")):
        ctx = make_ctx()
        run = A.run_agent(ctx, GeminiClient("fake-model", client=FakeClient([exc]), timeout_seconds=45))
        assert run.status == "rules_only_fallback" and run.fallback_reason == "api_failure" and run.proposal.source == "rules_only" and run.fallback_validation["ok"]
        assert "TIMEOUT after the configured 45s (45000 ms)" in run.fallback_detail
        assert any("RULES-ONLY FALLBACK: AGENT UNAVAILABLE (API FAILURE)" in l for l in run.labels) and any(l.startswith("APPROVAL REQUIRED") for l in run.labels)
        assert run.mode == "mock"  # a fake transport is never reported as live


# ------------------------------------------------------------------------------------------------ rate limits (the first live attempt hit the free tier's 5 requests/minute)

QUOTA_TEXT = ("429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota ... Generate content requests per minute limit exceeded on Free Tier. "
              "Please retry in 49.824436007s.'}}")


class Quota429(Exception):
    code = 429


class FakeClock:
    """Time that only moves when the limiter sleeps, so nothing in these tests really waits."""

    def __init__(self):
        self.now_s, self.sleeps = 1000.0, []

    def now(self):
        return self.now_s

    def sleep(self, s):
        self.sleeps.append(round(s, 3))
        self.now_s += s


def limiter(clock, **kw):
    from src.agent.llm import RateLimiter
    return RateLimiter(sleep=clock.sleep, clock=clock.now, **kw)


DAILY_TEXT = ("429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, "
              "head to: https://aistudio.google.com/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \\n* Quota exceeded for metric: "
              "generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-3.6-flash, description: Generate content requests per day limit exceeded on "
              "Free Tier. See https://aistudio.google.com/docs/rate-limits.\\nPlease retry in 16h59m45.837709624s.', 'status': 'RESOURCE_EXHAUSTED'}}")  # the real attempt-4 wording


class StructuredQuotaError(Exception):
    """Like the SDK's APIError: `.code` and `.details` (the response JSON), with the quota kind ONLY in the structured details, not in the message text."""

    def __init__(self, quota_id, retry_delay=None, message="429 RESOURCE_EXHAUSTED. quota exceeded"):
        details = [{"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests", "quotaId": quota_id}]}]
        if retry_delay:
            details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay})
        super().__init__(message)
        self.code, self.details = 429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": details}}


def test_rate_limit_errors_are_recognised_and_only_they_are_retried():
    from src.agent.llm import rate_limit_delay
    assert rate_limit_delay(Quota429(QUOTA_TEXT)) == 49.824436007
    assert rate_limit_delay(Quota429("slow down")) == 30.0 and rate_limit_delay(RuntimeError("RESOURCE_EXHAUSTED: quota")) == 30.0 and rate_limit_delay(RuntimeError("429 Too Many Requests")) == 30.0
    for not_a_rate_limit in (RuntimeError("500 internal error"), RuntimeError("401 API key not valid"), TimeoutError("timed out"), ValueError("bad"), RuntimeError("code 429 elsewhere in text")):
        assert rate_limit_delay(not_a_rate_limit) is None
    assert rate_limit_delay(Quota429(DAILY_TEXT)) is None  # a DAILY quota has no delay worth waiting for


@pytest.mark.parametrize("text,seconds", [("16h59m45.837709624s", 16 * 3600 + 59 * 60 + 45.837709624), ("49.8s", 49.8), ("2m30s", 150.0), ("1h", 3600.0), ("1h2m3s", 3723.0), ("59m", 3540.0),
                                          ("0.5s", 0.5), ("90s", 90.0), (" 3h ", 10800.0)])
def test_provider_durations_with_hours_minutes_and_seconds_are_parsed_correctly(text, seconds):
    from src.agent.llm import parse_duration_seconds
    assert parse_duration_seconds(text) == pytest.approx(seconds)


@pytest.mark.parametrize("text", ["", "soon", "5", "10ms", "h", "1d", "s5", "5 apples"])
def test_unparseable_durations_are_none_not_a_guess(text):
    from src.agent.llm import parse_duration_seconds
    assert parse_duration_seconds(text) is None


def test_daily_and_per_minute_limits_are_told_apart_from_structured_details_and_from_explicit_text():
    from src.agent.llm import classify_rate_limit
    real = classify_rate_limit(Quota429(DAILY_TEXT))
    assert (real.kind, round(real.retry_after, 3), real.source) == ("daily", 61185.838, "text")  # the real attempt-4 message, hours/minutes/seconds read correctly
    minute = classify_rate_limit(Quota429(QUOTA_TEXT))
    assert minute.kind == "per_minute" and minute.retry_after == pytest.approx(49.824436007)
    s = classify_rate_limit(StructuredQuotaError("GenerateRequestsPerDayPerProjectPerModel-FreeTier", "61185s"))
    assert (s.kind, s.source, s.retry_after) == ("daily", "structured", 61185.0)  # decided from the quota id even though the message text says nothing about a day
    m = classify_rate_limit(StructuredQuotaError("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "34s"))
    assert (m.kind, m.source, m.retry_after) == ("per_minute", "structured", 34.0)
    assert classify_rate_limit(StructuredQuotaError("GenerateRequestsPerDayPerProjectPerModel-FreeTier")).kind == "daily"  # no retry duration at all: still daily
    assert classify_rate_limit(Quota429("429 RESOURCE_EXHAUSTED daily limit reached")).kind == "daily" and classify_rate_limit(Quota429("slow down")).kind == "unspecified"
    assert classify_rate_limit(RuntimeError("500 internal")) is None
    both = StructuredQuotaError("GenerateRequestsPerMinutePerProjectPerModel-FreeTier")
    both.details["error"]["details"][0]["violations"].append({"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"})
    assert classify_rate_limit(both).kind == "daily"  # if a daily quota is among the violations, waiting a minute cannot help


def test_requests_are_paced_across_the_shared_limiter():
    clock = FakeClock()
    lim = limiter(clock, min_interval=13)
    for _ in range(3):
        lim.call(lambda: "ok")
    assert clock.sleeps == [13.0, 13.0]  # no wait before the first request, then one interval before each later one
    clock.now_s += 20  # a long gap: no extra wait
    lim.call(lambda: "ok")
    assert clock.sleeps == [13.0, 13.0]


def test_a_429_is_retried_after_the_delay_the_server_asked_for_and_the_run_continues():
    clock = FakeClock()
    fake = FakeClient([Quota429(QUOTA_TEXT), text_response('{"summary": "x", "items": []}')])
    lim = limiter(clock, max_retries=2, max_wait=90)
    turn = GeminiClient("m", client=fake, limiter=lim).generate("S", [{"role": "user", "text": "x"}], decls())
    assert turn.text == '{"summary": "x", "items": []}' and len(fake.calls) == 2 and clock.sleeps == [50.824] and lim.retries_used == 1  # 49.82 s asked + 1 s margin


def test_the_retry_is_bounded_and_a_persistent_429_takes_the_labelled_api_failure_fallback():
    clock = FakeClock()
    fake = FakeClient([Quota429(QUOTA_TEXT)] * 3)
    run = A.run_agent(make_ctx(), GeminiClient("m", client=fake, limiter=limiter(clock, max_retries=2, max_wait=90)))
    assert len(fake.calls) == 3 and len(clock.sleeps) == 2  # the original request plus exactly two retries, then it gives up
    assert run.status == "rules_only_fallback" and run.fallback_reason == "api_failure" and run.proposal.source == "rules_only"
    assert "RATE LIMIT (HTTP 429) still in force after 2 bounded retries" in run.fallback_detail and run.fallback_validation["ok"]
    assert any("RULES-ONLY FALLBACK: AGENT UNAVAILABLE (API FAILURE)" in l for l in run.labels)


def test_a_long_server_requested_wait_such_as_a_daily_quota_is_not_waited_for():
    clock = FakeClock()
    daily = Quota429("429 RESOURCE_EXHAUSTED. quota exceeded. Please retry in 3600s.")
    fake = FakeClient([daily])
    with pytest.raises(LLMError, match="RATE LIMIT .* after 0 bounded retries"):
        GeminiClient("m", client=fake, limiter=limiter(clock, max_wait=90)).generate("S", [{"role": "user", "text": "x"}], decls())
    assert clock.sleeps == [] and len(fake.calls) == 1


def test_a_daily_quota_error_makes_one_request_attempt_and_no_retry_sleep():
    """REGRESSION (independent review): 'requests per day limit exceeded ... retry in 16h59m45.8s' used to be treated as a 30-second delay: three attempts and two 31-second sleeps."""
    clock = FakeClock()
    fake = FakeClient([Quota429(DAILY_TEXT), text_response("{}"), text_response("{}")])  # later responses exist, so an (undesired) retry WOULD succeed
    lim = limiter(clock, max_retries=2, max_wait=90)
    with pytest.raises(LLMError) as exc:
        GeminiClient("m", client=fake, limiter=lim).generate("S", [{"role": "user", "text": "x"}], decls())
    assert len(fake.calls) == 1 and clock.sleeps == [] and lim.retries_used == 0
    assert "DAILY QUOTA EXHAUSTED (HTTP 429): no retry was attempted" in str(exc.value) and "17.0 h" in str(exc.value)  # 16h59m45s read as about 17 hours, not as seconds


@pytest.mark.parametrize("error", [
    Quota429("429 RESOURCE_EXHAUSTED. Generate content requests per day limit exceeded on Free Tier."),  # no retry duration at all
    Quota429("429 RESOURCE_EXHAUSTED. Generate content requests per day limit exceeded. Please retry in 5s."),  # a SHORT stated duration must not be believed for a daily quota
    Quota429("429 RESOURCE_EXHAUSTED. daily limit. retry in soon-ish"),  # unparseable duration
    StructuredQuotaError("GenerateRequestsPerDayPerProjectPerModel-FreeTier"),  # daily only in the structured details
    StructuredQuotaError("GenerateRequestsPerDayPerProjectPerModel-FreeTier", "2s"),
])
def test_a_daily_quota_stops_immediately_whether_or_not_its_retry_duration_can_be_read(error):
    clock = FakeClock()
    fake = FakeClient([error, text_response("{}")])
    with pytest.raises(LLMError, match="DAILY QUOTA EXHAUSTED"):
        GeminiClient("m", client=fake, limiter=limiter(clock, max_retries=5, max_wait=10_000)).generate("S", [{"role": "user", "text": "x"}], decls())
    assert len(fake.calls) == 1 and clock.sleeps == []


def test_a_delay_beyond_the_configured_maximum_stops_immediately_even_for_a_per_minute_limit():
    clock = FakeClock()
    fake = FakeClient([Quota429("429 RESOURCE_EXHAUSTED. Generate content requests per minute limit exceeded. Please retry in 2m1s."), text_response("{}")])  # 121 s > 90 s
    with pytest.raises(LLMError, match="RATE LIMIT .* after 0 bounded retries"):
        GeminiClient("m", client=fake, limiter=limiter(clock, max_retries=2, max_wait=90)).generate("S", [{"role": "user", "text": "x"}], decls())
    assert len(fake.calls) == 1 and clock.sleeps == []
    fake2 = FakeClient([StructuredQuotaError("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "95s"), text_response("{}")])  # structured retry delay beyond the maximum
    with pytest.raises(LLMError):
        GeminiClient("m", client=fake2, limiter=limiter(FakeClock(), max_wait=90)).generate("S", [{"role": "user", "text": "x"}], decls())
    assert len(fake2.calls) == 1


def test_short_per_minute_throttling_keeps_its_bounded_retries_including_a_structured_one():
    clock = FakeClock()
    fake = FakeClient([StructuredQuotaError("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "34s"), Quota429("429 RESOURCE_EXHAUSTED per minute. Please retry in 1m2s."), text_response('{"summary": "x", "items": []}')])
    lim = limiter(clock, max_retries=2, max_wait=90)
    turn = GeminiClient("m", client=fake, limiter=lim).generate("S", [{"role": "user", "text": "x"}], decls())
    assert turn.text and len(fake.calls) == 3 and clock.sleeps == [35.0, 63.0] and lim.retries_used == 2  # 34 s and 62 s asked, plus the 1 s margin each
    third = FakeClient([Quota429(QUOTA_TEXT)] * 4)
    with pytest.raises(LLMError, match="after 2 bounded retries"):
        GeminiClient("m", client=third, limiter=limiter(FakeClock(), max_retries=2, max_wait=90)).generate("S", [{"role": "user", "text": "x"}], decls())
    assert len(third.calls) == 3  # still bounded


BILLING_TEXT = ("402 RESOURCE_EXHAUSTED. {'error': {'code': 402, 'message': 'Your prepayment credits are depleted. Please go to AI Studio at https://ai.studio/projects to manage your "
                "project and billing. Learn more at https://ai.google.dev/gemini-api/docs/billing#prepay. ', 'status': 'RESOURCE_EXHAUSTED'}}")  # the real attempt-5 error


class Billing402(Exception):
    code = 402


def test_a_402_credits_depleted_error_is_a_billing_state_not_a_rate_limit_and_is_never_retried():
    """REGRESSION: attempt 5's HTTP 402 'prepayment credits are depleted' carries the status text RESOURCE_EXHAUSTED, so it was mislabelled 'RATE LIMITED (HTTP 429)' and retried twice."""
    from src.agent.llm import classify_rate_limit, rate_limit_delay
    assert classify_rate_limit(Billing402(BILLING_TEXT)).kind == "billing"
    assert classify_rate_limit("API error: ClientError: " + BILLING_TEXT).kind == "billing"  # the connectivity check records only text
    assert rate_limit_delay(Billing402(BILLING_TEXT)) is None
    # an explicit non-429 code is not a rate limit even if its status text says RESOURCE_EXHAUSTED
    class Other(Exception):
        code = 503
    assert classify_rate_limit(Other("503 RESOURCE_EXHAUSTED overloaded")) is None and classify_rate_limit("ServerError: 504 DEADLINE_EXCEEDED. deadline expired") is None
    clock = FakeClock()
    fake = FakeClient([Billing402(BILLING_TEXT), text_response("{}"), text_response("{}")])
    lim = limiter(clock, max_retries=5, max_wait=10_000)
    with pytest.raises(LLMError) as exc:
        GeminiClient("m", client=fake, limiter=lim).generate("S", [{"role": "user", "text": "x"}], decls())
    assert len(fake.calls) == 1 and clock.sleeps == [] and lim.retries_used == 0
    assert "BILLING" in str(exc.value) and "HTTP 402" in str(exc.value) and "not a rate limit" in str(exc.value) and "RATE LIMIT (HTTP 429)" not in str(exc.value)
    run = A.run_agent(make_ctx(), GeminiClient("m", client=FakeClient([Billing402(BILLING_TEXT)]), limiter=limiter(FakeClock())))
    assert run.status == "rules_only_fallback" and run.fallback_reason == "api_failure" and "BILLING" in run.fallback_detail and run.fallback_validation["ok"]


def test_the_connectivity_check_reports_a_billing_state_as_such_and_stops_for_the_whole_project():
    from src.agent.live import select_verified_model
    from tests.test_agent_live_selection import FakeApi

    class BillingApi(FakeApi):
        def generate_content(self, model, contents, config=None):
            self.used.append(model)
            raise Billing402(BILLING_TEXT)

    api = BillingApi(["models/gemini-x-flash", "models/gemini-y-flash-lite"], good=[])
    sel = select_verified_model(api)
    assert sel["verified_model"] is None and sel["stopped"].startswith("BILLING (HTTP 402)") and "every model on this key's project" in sel["stopped"] and len(api.used) == 1
    assert "RATE LIMITED" not in sel["stopped"]


def test_a_daily_quota_during_a_run_takes_the_labelled_fallback_without_waiting():
    clock = FakeClock()
    fake = FakeClient([Quota429(DAILY_TEXT)])
    run = A.run_agent(make_ctx(), GeminiClient("m", client=fake, limiter=limiter(clock)))
    assert run.status == "rules_only_fallback" and run.fallback_reason == "api_failure" and "DAILY QUOTA EXHAUSTED" in run.fallback_detail and clock.sleeps == [] and len(fake.calls) == 1
    assert run.proposal.source == "rules_only" and run.fallback_validation["ok"]


def test_the_connectivity_check_names_a_daily_quota_and_stops():
    from src.agent.live import select_verified_model
    from tests.test_agent_live_selection import FakeApi

    class DailyApi(FakeApi):
        def generate_content(self, model, contents, config=None):
            self.used.append(model)
            raise Quota429(DAILY_TEXT)

    api = DailyApi(["models/gemini-x-flash", "models/gemini-y-flash"], good=[])
    sel = select_verified_model(api)
    assert sel["verified_model"] is None and sel["stopped"].startswith("DAILY QUOTA EXHAUSTED") and "17.0 h" in sel["stopped"] and len(api.used) == 1


def test_other_errors_are_never_retried():
    clock = FakeClock()
    fake = FakeClient([RuntimeError("500 internal error"), text_response("{}")])
    with pytest.raises(LLMError, match="RuntimeError"):
        GeminiClient("m", client=fake, limiter=limiter(clock)).generate("S", [{"role": "user", "text": "x"}], decls())
    assert len(fake.calls) == 1 and clock.sleeps == []


def test_model_selection_and_the_agent_share_one_limiter_and_the_connectivity_check_survives_a_429():
    from src.agent.live import select_verified_model
    from src.agent.llm import PacedClient
    from tests.test_agent_live_selection import FakeApi

    class FlakyApi(FakeApi):
        flaked = False

        def generate_content(self, model, contents, config=None):
            if not self.flaked:
                self.flaked = True
                raise Quota429(QUOTA_TEXT)
            return super().generate_content(model, contents, config)

    clock = FakeClock()
    lim = limiter(clock, min_interval=13, max_retries=2, max_wait=90)
    sel = select_verified_model(PacedClient(FlakyApi(["models/gemini-x-flash"], good=["models/gemini-x-flash"]), lim))
    assert sel["verified_model"] == "models/gemini-x-flash" and lim.retries_used == 1
    waited = len(clock.sleeps)
    agent_client = GeminiClient("m", client=FakeClient([text_response("{}")]), limiter=lim)
    agent_client.generate("S", [{"role": "user", "text": "x"}], decls())
    assert len(clock.sleeps) == waited + 1 and clock.sleeps[-1] >= 0  # the agent's first request was paced against the connectivity check's last one


def test_the_rate_limit_settings_come_from_the_agent_configuration_and_reach_the_live_path():
    import yaml
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "config" / "agent.yaml").read_text(encoding="utf-8"))
    assert cfg["min_seconds_between_requests"] >= 12 and cfg["max_rate_limit_retries"] == 2 and cfg["max_rate_limit_wait_seconds"] == 90  # free tier: 5 requests/minute
    src = (root / "scripts" / "run_agent.py").read_text(encoding="utf-8")
    assert 'cfg.get("min_seconds_between_requests"' in src and "PacedClient(client, limiter)" in src and "limiter=limiter" in src


def test_the_tool_call_bound_still_fits_a_complete_investigation_of_every_channel_with_planning_and_a_repair():
    """The second live attempt made overview + 4 per-channel tools x 5 channels = 21 calls and hit the old bound of 20. The bound stays hard, but must fit this
    thorough pattern plus planning for every channel, a constraint check and a repair's worth of re-checks."""
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "agent.yaml").read_text(encoding="utf-8"))
    channels = len(make_ctx().channels)
    sweep = 1 + 4 * channels
    assert sweep == 21 and cfg["max_tool_calls"] >= sweep + channels + 1 + 4 and cfg["max_tool_calls"] <= 60  # generous, but still a bound
    assert cfg["max_model_turns"] >= 10 and cfg["repair_retries"] == 1


def test_a_complete_per_channel_sweep_fits_the_bound_and_the_old_bound_ran_out_on_it():
    ctx = make_ctx()
    calls = [("get_review_overview", {})] + [(t, {"channel": c}) for t in ("get_policy_decision", "get_quality_checks", "get_forecast_and_populations", "get_economics") for c in ctx.channels]
    from src.agent.llm import MockLLM, call_turn
    _, good = standard_conversation(ctx)
    old = A.run_agent(ctx, MockLLM([call_turn(*calls), good[1]]), max_tool_calls=20)  # the first two live attempts' bound
    assert len([c for c in old.tool_calls if c["status"] != "denied"]) == 20 and [c["status"] for c in old.tool_calls].count("denied") == 1
    assert any(l.startswith("TOOL BUDGET REACHED: 20 of 20") for l in old.labels)  # the 21st call was refused (the old code fell back here without letting the model answer)
    new = A.run_agent(make_ctx(), MockLLM([call_turn(*calls), call_turn(("plan_experiment", {"channel": "applovin", "new_budget": 110.0})), good[1]]), max_tool_calls=40)
    assert len(new.tool_calls) == 22 and not any(c["status"] == "denied" for c in new.tool_calls) and not any(l.startswith("TOOL BUDGET REACHED") for l in new.labels)


def test_model_listing_and_candidate_ordering_use_the_listing_only():
    assert list_generate_models(FakeClient([])) == ["models/gemini-x-flash"]  # embedding-only models are not offered
    listing = ["models/gemini-9-pro", "models/gemini-9-flash-lite-preview", "models/gemini-9-flash", "models/gemini-9-flash-image", "models/gemma-3"]
    order = candidate_models(listing)
    assert order[0] == "models/gemini-9-flash" and order.index("models/gemini-9-flash-lite-preview") > order.index("models/gemini-9-pro")
    assert set(order) == set(listing)  # nothing is invented or dropped: the caller verifies each candidate with the live round-trip check
