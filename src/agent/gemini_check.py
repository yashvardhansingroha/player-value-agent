"""Gemini connectivity check logic (importable so tests run the SAME code the script runs).

`run_check(client, model)` performs three steps and validates each one:
  1. plain call: the reply must be "ok" (case/whitespace-insensitive)
  2. tool call: exactly one add_numbers call with numeric args {a, b} = {2, 3} (either order)
  3. round trip: run the local function, send the result back using the SDK's conversation
     format (original model content preserved, FunctionResponse carries the call id), and
     require a final text answer that reports 5.

`client` is anything with `client.models.generate_content(...)`; tests pass a fake one.
Nothing here reads credentials or the network by itself.
"""
import math
import re
from dataclasses import dataclass, field

from google.genai import types

PLAIN_PROMPT = "Reply with the single word: ok"
TOOL_PROMPT = "Use the add_numbers tool to add 2 and 3. Do not do the arithmetic yourself."
TOOL_NAME = "add_numbers"
EXPECTED_VALUES = [2, 3]
EXPECTED_SUM = 5


def add_numbers(a, b):
    return a + b


ADD_NUMBERS_DECLARATION = types.FunctionDeclaration(
    name=TOOL_NAME,
    description="Add two numbers and return the sum.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={"a": types.Schema(type=types.Type.NUMBER), "b": types.Schema(type=types.Type.NUMBER)},
        required=["a", "b"],
    ),
)


def tool_config() -> types.GenerateContentConfig:
    return types.GenerateContentConfig(tools=[types.Tool(function_declarations=[ADD_NUMBERS_DECLARATION])])


@dataclass
class CheckResult:
    steps: list = field(default_factory=list)  # (step name, passed, detail)

    def add(self, name, passed, detail=""):
        self.steps.append((name, passed, detail))
        return passed

    @property
    def ok(self):
        return bool(self.steps) and all(passed for _, passed, _ in self.steps)

    @property
    def plain_ok(self):
        return self._step_ok("plain call")

    @property
    def tool_call_ok(self):
        return self._step_ok("tool call")

    @property
    def round_trip_ok(self):
        return self._step_ok("round trip")

    def _step_ok(self, name):
        return any(n == name and passed for n, passed, _ in self.steps)


def check_plain_text(text):
    """Return (passed, detail). Only 'ok' (ignoring whitespace/case) passes."""
    if text is None or not text.strip():
        return False, "empty plain response"
    if text.strip().lower() != "ok":
        return False, f"unexpected plain response: {text.strip()[:80]!r}"
    return True, "plain response was 'ok'"


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check_tool_call(function_calls):
    """Return (passed, detail, call). Exactly one add_numbers call with a, b numeric = 2 and 3."""
    calls = list(function_calls or [])
    if len(calls) == 0:
        return False, "no tool call returned", None
    if len(calls) != 1:
        return False, f"expected exactly 1 tool call, got {len(calls)}", None
    call = calls[0]
    if call.name != TOOL_NAME:
        return False, f"wrong function name: {call.name!r}", None
    args = call.args
    if not isinstance(args, dict):
        return False, "tool call arguments missing or malformed", None
    if set(args) != {"a", "b"}:
        return False, f"arguments must be exactly a and b, got {sorted(args)}", None
    if not (_is_number(args["a"]) and _is_number(args["b"])):
        return False, "arguments a and b must both be numbers", None
    if sorted([args["a"], args["b"]]) != EXPECTED_VALUES:
        return False, f"expected values 2 and 3, got a={args['a']}, b={args['b']}", None
    return True, "exactly one add_numbers call with a, b = 2, 3", call


# a number not glued to letters/digits; a trailing sentence period ("is 5.") is allowed, a decimal point ("5.5") is not split
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?!\w|\.\d)")


def check_final_answer(text):
    """Return (passed, detail). The reply must report 5 (digit or the word five) and no other result."""
    if text is None or not text.strip():
        return False, "empty final response"
    numbers = [float(n) for n in _NUMBER.findall(text)]
    says_five = EXPECTED_SUM in numbers or re.search(r"\bfive\b", text, re.IGNORECASE) is not None
    # 2 and 3 may legitimately be repeated back ("2 + 3 = 5"); any other number is a wrong answer
    stray = [n for n in numbers if n not in (2.0, 3.0, 5.0)]
    if not says_five:
        return False, f"final response does not report 5: {text.strip()[:80]!r}"
    if stray:
        return False, f"final response contains other numbers {stray}: {text.strip()[:80]!r}"
    return True, "final response reports 5"


def run_check(client, model, add_fn=add_numbers) -> CheckResult:
    """Run all three steps against `client`. Any API exception is recorded as a failed step."""
    result = CheckResult()

    # Step 1: plain call
    try:
        reply = client.models.generate_content(model=model, contents=PLAIN_PROMPT)
        passed, detail = check_plain_text(reply.text)
    except Exception as exc:  # network/API failure must be reported, not crash
        passed, detail = False, f"API error: {type(exc).__name__}: {exc}"
    if not result.add("plain call", passed, detail):
        return result

    # Step 2: tool call
    config = tool_config()
    user_turn = types.Content(role="user", parts=[types.Part(text=TOOL_PROMPT)])
    try:
        reply = client.models.generate_content(model=model, contents=[user_turn], config=config)
        passed, detail, call = check_tool_call(reply.function_calls)
    except Exception as exc:
        passed, detail, call = False, f"API error: {type(exc).__name__}: {exc}", None
    if not result.add("tool call", passed, detail):
        return result

    # Step 3: round trip. Keep the model's own content unchanged (preserves call ids and any thought signatures).
    try:
        candidates = reply.candidates or []
        model_content = candidates[0].content if candidates else None
        if model_content is None:
            result.add("round trip", False, "model response had no content to send back")
            return result
        tool_output = add_fn(call.args["a"], call.args["b"])
        function_response = types.Part(
            function_response=types.FunctionResponse(id=call.id, name=call.name, response={"result": tool_output})
        )
        history = [user_turn, model_content, types.Content(role="user", parts=[function_response])]
        final = client.models.generate_content(model=model, contents=history, config=config)
        if final.function_calls:
            passed, detail = False, "model asked for another tool call instead of answering"
        else:
            passed, detail = check_final_answer(final.text)
    except Exception as exc:
        passed, detail = False, f"API error: {type(exc).__name__}: {exc}"
    result.add("round trip", passed, detail)
    return result
