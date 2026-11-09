"""Deterministic tests for the Gemini check. ALL RESPONSES HERE ARE MOCKED: they exercise the production
checker code (src/agent/gemini_check.py) but say nothing about live Gemini connectivity.
No credentials or network are used."""
import os
import subprocess
import sys
from pathlib import Path

import pytest
from google.genai import types

from src.agent.gemini_check import run_check

ROOT = Path(__file__).resolve().parents[1]


def text_response(text):
    parts = [types.Part(text=text)] if text is not None else []
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=parts))]
    )


def call_response(calls):
    """calls: list of (name, args, id)."""
    parts = [types.Part(function_call=types.FunctionCall(name=n, args=a, id=i)) for n, a, i in calls]
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=parts))]
    )


class FakeClient:
    """Stands in for genai.Client. `script` items are responses, exceptions, or callables(contents)->response."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []  # recorded kwargs of every generate_content call
        self.models = self

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if callable(item):
            return item(kwargs["contents"])
        return item


def echo_sum_from_history(contents):
    """A fake model that can only answer correctly if the checker really sent the tool result back."""
    for content in contents:
        for part in content.parts or []:
            if part.function_response is not None:
                return text_response(f"The sum is {part.function_response.response['result']}.")
    raise AssertionError("tool result was not sent back to the model")


GOOD_CALL = [("add_numbers", {"a": 2, "b": 3}, "call-1")]


def test_mocked_full_round_trip_success_and_result_sent_back():
    model_turn = call_response([("add_numbers", {"a": 3, "b": 2}, "call-7")])  # swapped order is accepted
    client = FakeClient([text_response(" OK \n"), model_turn, echo_sum_from_history])
    seen = []

    def spy_add(a, b):
        seen.append((a, b))
        return a + b

    result = run_check(client, "mock-model", add_fn=spy_add)

    assert result.ok and result.plain_ok and result.tool_call_ok and result.round_trip_ok
    assert seen == [(3, 2)]  # the local function really ran with the model's arguments
    assert len(client.calls) == 3

    history = client.calls[2]["contents"]
    assert history[1] is model_turn.candidates[0].content  # model content preserved unchanged
    response_part = history[2].parts[0].function_response
    assert history[2].role == "user"
    assert response_part.id == "call-7" and response_part.name == "add_numbers"
    assert response_part.response == {"result": 5}
    assert client.calls[1]["config"].tools and client.calls[2]["config"].tools  # tools stay declared


@pytest.mark.parametrize("bad_text", ["yes", "okay then", "", "   ", None])
def test_mocked_incorrect_plain_response_fails_and_stops(bad_text):
    client = FakeClient([text_response(bad_text)])
    result = run_check(client, "mock-model")
    assert not result.ok and not result.plain_ok
    assert len(client.calls) == 1  # no tool call attempted after a failed plain check


@pytest.mark.parametrize(
    "calls",
    [
        [("subtract_numbers", {"a": 2, "b": 3}, "c1")],  # wrong function name
        [("add_numbers", {"a": 2}, "c1")],  # missing argument
        [("add_numbers", {}, "c1")],  # no arguments
        [("add_numbers", {"a": 2, "b": 4}, "c1")],  # incorrect value
        [("add_numbers", {"a": 2, "b": 3, "c": 1}, "c1")],  # additional argument
        [("add_numbers", {"a": "2", "b": "3"}, "c1")],  # malformed: strings
        [("add_numbers", {"a": True, "b": 3}, "c1")],  # malformed: bool is not a number here
        [("add_numbers", {"a": 2, "b": 3}, "c1"), ("add_numbers", {"a": 2, "b": 3}, "c2")],  # extra tool call
        [("add_numbers", {"a": 2, "b": 3}, "c1"), ("other_tool", {}, "c2")],  # unexpected extra call
    ],
)
def test_mocked_bad_tool_call_fails_without_round_trip(calls):
    client = FakeClient([text_response("ok"), call_response(calls)])
    result = run_check(client, "mock-model")
    assert result.plain_ok and not result.tool_call_ok and not result.round_trip_ok and not result.ok
    assert len(client.calls) == 2  # never sent a tool result back


def test_mocked_missing_tool_call_fails():
    client = FakeClient([text_response("ok"), text_response("The answer is 5.")])
    result = run_check(client, "mock-model")
    assert not result.tool_call_ok and not result.ok
    assert "no tool call" in result.steps[-1][2]


@pytest.mark.parametrize(
    "final",
    [
        text_response("The sum is 6."),
        text_response("I could not compute it."),
        text_response("2 + 3 = 5, and also 7."),
        text_response(""),
        call_response(GOOD_CALL),  # asks for another tool call instead of answering
    ],
)
def test_mocked_incorrect_final_answer_fails(final):
    client = FakeClient([text_response("ok"), call_response(GOOD_CALL), final])
    result = run_check(client, "mock-model")
    assert result.plain_ok and result.tool_call_ok and not result.round_trip_ok and not result.ok


def test_mocked_wrong_tool_result_is_caught_end_to_end():
    client = FakeClient([text_response("ok"), call_response(GOOD_CALL), echo_sum_from_history])
    result = run_check(client, "mock-model", add_fn=lambda a, b: a + b + 1)  # tool returns 6
    assert not result.round_trip_ok and not result.ok


def test_mocked_accepts_word_five_and_repeated_operands():
    for final in ("Five.", "2 plus 3 equals 5.", "2 + 3 = 5"):
        client = FakeClient([text_response("ok"), call_response(GOOD_CALL), text_response(final)])
        assert run_check(client, "mock-model").ok, final


@pytest.mark.parametrize("fail_at", [0, 1, 2])
def test_mocked_api_failure_at_each_step(fail_at):
    script = [text_response("ok"), call_response(GOOD_CALL), echo_sum_from_history]
    script[fail_at] = ConnectionError("network down")
    result = run_check(FakeClient(script), "mock-model")
    assert not result.ok
    assert "API error: ConnectionError" in result.steps[-1][2]
    if fail_at < 2:
        assert not result.round_trip_ok


def test_no_key_skips_cleanly_without_network():
    env = {k: v for k, v in os.environ.items() if k != "GEMINI_API_KEY"}
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_gemini.py")],
        capture_output=True, text=True, env=env, cwd=ROOT, timeout=60,
    )
    assert done.returncode == 0
    assert "SKIPPED" in done.stdout and "UNVERIFIED" in done.stdout
    assert "LIVE" not in done.stdout
