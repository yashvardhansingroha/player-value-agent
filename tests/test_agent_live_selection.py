"""Model selection helper with a FAKE client: the candidate order comes from the listing, and a model counts as verified only if the plain / tool /
round-trip check passes. MOCKED: says nothing about live connectivity."""
from google.genai import types

from src.agent.live import select_verified_model


class FakeApi:
    """Behaves like a model that answers correctly for models named in `good` and fails for the others."""

    def __init__(self, listing, good):
        self.listing, self.good, self.models, self.used = listing, set(good), self, []

    def list(self):
        return [types.Model(name=n, supported_actions=["generateContent"]) for n in self.listing]

    def generate_content(self, model, contents, config=None):
        self.used.append(model)
        if model not in self.good:
            raise RuntimeError("404 model not found")
        parts_all = [p for c in (contents if isinstance(contents, list) else []) for p in (c.parts or [])]
        if any(p.function_response is not None for p in parts_all):
            return self._text("The sum is 5.")
        if config is not None and config.tools:
            return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model", parts=[
                types.Part(function_call=types.FunctionCall(name="add_numbers", args={"a": 2, "b": 3}, id="c1"))]))])
        return self._text("ok")

    @staticmethod
    def _text(t):
        return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part(text=t)]))])


def test_the_first_candidate_that_passes_all_three_steps_is_the_verified_model_and_failures_are_recorded():
    api = FakeApi(["models/gemini-x-flash-lite-preview", "models/gemini-x-pro", "models/gemini-x-flash"], good={"models/gemini-x-pro"})
    sel = select_verified_model(api)
    assert sel["listed"] == ["models/gemini-x-flash-lite-preview", "models/gemini-x-pro", "models/gemini-x-flash"]
    assert [t["model"] for t in sel["tried"]] == ["models/gemini-x-flash", "models/gemini-x-pro"]  # ranked from the listing; the failing one is tried first and recorded
    assert sel["tried"][0]["passed"] is False and sel["tried"][1]["passed"] is True and sel["verified_model"] == "models/gemini-x-pro"


def test_no_model_is_verified_if_every_candidate_fails_and_a_listing_failure_is_reported():
    sel = select_verified_model(FakeApi(["models/a", "models/b"], good=[]))
    assert sel["verified_model"] is None and len(sel["tried"]) == 2 and all(not t["passed"] for t in sel["tried"])

    class NoList(FakeApi):
        def list(self):
            raise RuntimeError("403 forbidden")
    out = select_verified_model(NoList([], []))
    assert out["verified_model"] is None and out["listed"] == [] and out["list_error"] == "RuntimeError"


def test_a_rate_limit_stops_the_selection_instead_of_spending_more_quota_and_keeps_the_full_message():
    long_message = "429 RESOURCE_EXHAUSTED. " + "You exceeded your current quota. " * 12 + "Quota exceeded for metric: generate_content_free_tier_requests, limit: 5. Please retry in 49.8s."

    class QuotaApi(FakeApi):
        def generate_content(self, model, contents, config=None):
            self.used.append(model)
            parts = [p for c in (contents if isinstance(contents, list) else []) for p in (c.parts or [])]
            if config is not None and config.tools and not any(p.function_response for p in parts):
                raise RuntimeError(long_message)  # the plain call works, the tool call hits the quota
            return super().generate_content(model, contents, config)

    api = QuotaApi(["models/gemini-x-flash", "models/gemini-y-flash", "models/gemini-z-flash"], good=["models/gemini-x-flash", "models/gemini-y-flash", "models/gemini-z-flash"])
    sel = select_verified_model(api)
    assert sel["verified_model"] is None and len(sel["tried"]) == 1 and "RATE LIMITED (HTTP 429)" in sel["stopped"] and "not the model" in sel["stopped"]
    tool_step = sel["tried"][0]["steps"][1]
    assert tool_step[0] == "tool call" and "Quota exceeded for metric" in tool_step[2] and "Please retry in 49.8s" in tool_step[2]  # the old 160-character cut hid this
    assert len(set(api.used)) == 1  # the other candidates were never called


def test_live_cases_can_be_run_one_at_a_time_to_fit_a_small_daily_quota():
    import pytest
    from scripts.run_agent import CASES, select_cases
    assert select_cases(None) == CASES and len(CASES) == 2
    assert [d for d, _ in select_cases(["2025-11-10"])] == ["2025-11-10"] and [d for d, _ in select_cases(["2025-11-10", "2025-10-13"])] == ["2025-10-13", "2025-11-10"]
    with pytest.raises(SystemExit, match="unknown case"):
        select_cases(["2025-01-01"])


def test_a_requested_model_is_verified_not_trusted():
    api = FakeApi(["models/gemini-x-flash"], good=[])
    sel = select_verified_model(api, requested="models/made-up")
    assert sel["verified_model"] is None and [t["model"] for t in sel["tried"]] == ["models/made-up"]
