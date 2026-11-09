"""LLM clients behind one tiny interface. Mock, replay and live results are kept strictly apart: every client has a `mode`
('mock' | 'replay' | 'live') and the agent run records it. ALL DATA IS SYNTHETIC.

    client.generate(system, history, tools) -> LLMTurn

`history` is a neutral list: {"role": "user", "text"} | {"role": "model", "turn": LLMTurn} | {"role": "tool", "results": [...]}.
A turn either asks for tool calls or carries final text. The model never executes anything: the agent loop does, through the
tool registry in tools.py.

  MockLLM     scripted turns for deterministic tests. Its results are plumbing tests and are NEVER agent evidence.
  ReplayLLM   plays back the model side of a recording saved from an actual LIVE run (the deterministic tools run again).
  GeminiClient  the live client (google-genai). The key is read ONLY from the GEMINI_API_KEY environment variable, is never stored
              and is redacted from every error message.
"""
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.ledger.proposal import jsonable

KEY_VAR = "GEMINI_API_KEY"


_DURATION = re.compile(r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m(?!s))?(?:(\d+(?:\.\d+)?)s)?")


def parse_duration_seconds(text):
    """Seconds in a provider duration such as '49.8s', '2m30s', '1h' or '16h59m45.837709624s'; None if `text` is not (only) such a duration."""
    m = _DURATION.fullmatch(str(text).strip().replace(" ", ""))
    if not m or not any(m.groups()):
        return None
    h, mi, s = (float(g) if g else 0.0 for g in m.groups())
    return h * 3600.0 + mi * 60.0 + s


@dataclass
class RateLimitInfo:
    kind: str  # 'daily' (a per-day quota is exhausted: waiting is pointless) | 'per_minute' | 'unspecified' (a 429 that says neither) | 'billing' (HTTP 402: prepaid credits depleted)
    retry_after: float = None  # seconds the provider asked us to wait, if it said and it could be parsed
    source: str = "text"  # 'structured' (the error's own quota details) | 'text' (the message wording)


def _error_details(exc) -> list:
    """The provider's structured error details (a list of dicts), from the SDK error's `details` (the response JSON, with or without an 'error' wrapper)."""
    d = getattr(exc, "details", None)
    if isinstance(d, dict):
        d = d.get("error", d).get("details", []) if isinstance(d.get("error", d), dict) else []
    return [x for x in d if isinstance(x, dict)] if isinstance(d, list) else []


def classify_rate_limit(exc_or_text):
    """None if this is not a rate-limit error (HTTP 429 / RESOURCE_EXHAUSTED). Otherwise a RateLimitInfo.
    DAILY exhaustion is recognised from the structured QuotaFailure violations (a quota id such as '...PerDay...') when the SDK exposes them, and otherwise from
    explicit daily wording ('requests per day', 'daily'); it is decided WITHOUT needing the retry duration to be parseable. A per-day quota resets in hours, so
    the caller must not retry it. Everything else that is a 429 is a temporary limit. The retry duration comes from RetryInfo.retryDelay or 'retry in <duration>'."""
    text = str(exc_or_text)
    code = getattr(exc_or_text, "code", None)
    if not isinstance(code, int):  # plain text (for example the connectivity check's recorded message): read the status code from it, e.g. 'ClientError: 402 RESOURCE_EXHAUSTED'
        m = re.search(r"(?:^|[:\s])([1-5]\d\d)\s+[A-Z][A-Z_]+", text)
        code = int(m.group(1)) if m else None
    if code == 402 or re.search(r"prepayment credits? (?:are |is )?depleted", text, re.IGNORECASE):
        # NOT a rate limit: the project has no credits (billing state). It affects every model on the project and no wait fixes it.
        return RateLimitInfo("billing", None, "structured" if code == 402 else "text")
    if code is not None and code != 429:
        return None  # an explicit status code that is not 429 is not a rate limit, whatever its status text says
    if code is None and "RESOURCE_EXHAUSTED" not in text and not text.lstrip().startswith("429"):
        return None
    kind, source, retry = None, "text", None
    for d in _error_details(exc_or_text):
        t = str(d.get("@type", ""))
        if t.endswith("QuotaFailure"):
            for v in d.get("violations", []) or []:
                ident = re.sub(r"[^a-z]", "", " ".join(str(v.get(k, "")) for k in ("quotaId", "quotaMetric", "description")).lower())
                if "perday" in ident or "daily" in ident:
                    kind, source = "daily", "structured"
                elif "perminute" in ident and kind is None:
                    kind, source = "per_minute", "structured"
        elif t.endswith("RetryInfo") and retry is None:
            retry = parse_duration_seconds(d.get("retryDelay", ""))
    if kind != "daily" and re.search(r"per\s*day|\bdaily\b|requests\s+per\s+day", text, re.IGNORECASE):
        kind, source = "daily", "text"
    if kind is None:
        kind = "per_minute" if re.search(r"per\s*minute", text, re.IGNORECASE) else "unspecified"
    if retry is None:
        m = re.search(r"retry in\s+([0-9hms.]+)", text, re.IGNORECASE)
        retry = parse_duration_seconds(m.group(1).rstrip(".")) if m else None
    return RateLimitInfo(kind, retry, source)


def rate_limit_delay(exc):
    """Compatibility helper: seconds to wait before retrying a TEMPORARY rate limit (30 s when none is stated), or None if `exc` is not a rate limit or is a DAILY one."""
    info = classify_rate_limit(exc)
    if info is None or info.kind in ("daily", "billing"):
        return None
    return info.retry_after if info.retry_after is not None else 30.0


class RateLimiter:
    """Request pacing plus a BOUNDED retry for rate-limit errors, shared by every call that uses it (model selection and the agent).
      min_interval  minimum seconds between the starts of two requests (the free tier allows 5 per minute, i.e. 12 s; 0 disables pacing)
      max_retries   how many times one request may be retried after a 429; max_wait: if the server asks for longer than this, give up at once
    So the worst case is bounded: max_retries x (max_wait + 1 s) of waiting per request, after which the caller gets the error (and the agent its labelled fallback)."""

    def __init__(self, min_interval: float = 0.0, max_retries: int = 2, max_wait: float = 90.0, sleep=time.sleep, clock=time.monotonic):
        self.min_interval, self.max_retries, self.max_wait, self._sleep, self._clock = min_interval, max_retries, max_wait, sleep, clock
        self._last, self.retries_used = None, 0

    def call(self, fn):
        attempt = 0
        while True:
            if self._last is not None and self.min_interval > 0:
                wait = self.min_interval - (self._clock() - self._last)
                if wait > 0:
                    self._sleep(wait)
            self._last = self._clock()
            try:
                return fn()
            except Exception as exc:
                info = classify_rate_limit(exc)
                if info is None:
                    raise  # not a rate limit: never retried
                delay = info.retry_after if info.retry_after is not None else 30.0
                stop = info.kind in ("daily", "billing") or attempt >= self.max_retries or delay > self.max_wait  # a DAILY quota or a BILLING state stops at once  # a DAILY quota stops at once, whatever its stated duration
                try:
                    exc.rate_limit_kind, exc.rate_limit_retry_after = info.kind, info.retry_after  # lets the caller label the failure
                    if stop:
                        exc.rate_limit_retries = attempt
                except AttributeError:
                    pass
                if stop:
                    raise
                attempt += 1
                self.retries_used += 1
                self._sleep(delay + 1.0)


class PacedClient:
    """Wraps a raw SDK client so that models.list / models.generate_content go through a shared RateLimiter (used for the connectivity check)."""

    def __init__(self, client, limiter: RateLimiter):
        self._client, self._limiter, self.models = client, limiter, self

    def list(self):
        return self._limiter.call(lambda: list(self._client.models.list()))

    def generate_content(self, **kwargs):
        return self._limiter.call(lambda: self._client.models.generate_content(**kwargs))


class LLMError(Exception):
    """The model could not be reached or answered with an API error. The message is already redacted."""


def http_timeout_ms(seconds) -> int:
    """The agent configuration states seconds; the installed google-genai SDK documents its HTTP timeout in MILLISECONDS. Convert explicitly."""
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not 0 < seconds < float("inf"):
        raise ValueError(f"timeout_seconds must be a positive number, got {seconds!r}")
    return int(round(seconds * 1000))


@dataclass
class ToolCall:
    name: str
    args: dict
    provider_id: str = None  # the provider's own call id (sent back with the result); not the evidence id


@dataclass
class LLMTurn:
    text: str = None
    tool_calls: list = field(default_factory=list)
    raw: object = None  # provider object sent back unchanged on the next request (keeps thought signatures); never recorded
    usage: dict = None

    def as_record(self) -> dict:
        return {"text": self.text, "tool_calls": [{"name": c.name, "args": jsonable(c.args)} for c in self.tool_calls]}


# ---------------------------------------------------------------------------------------------- mock


def call_turn(*calls) -> LLMTurn:
    """Mock helper: a turn that asks for tool calls, each given as (name, args dict)."""
    return LLMTurn(tool_calls=[ToolCall(n, dict(a), f"mock-{i}") for i, (n, a) in enumerate(calls)])


def text_turn(text: str) -> LLMTurn:
    return LLMTurn(text=text)


class MockLLM:
    """Scripted model. Items are LLMTurn, an Exception to raise (simulated API failure), or callable(history) -> LLMTurn."""
    mode = "mock"
    provider = "mock"
    model_id = "mock-llm (scripted)"

    def __init__(self, script):
        self.script, self.requests = list(script), []

    def generate(self, system, history, tools):
        self.requests.append({"system": system, "history": list(history), "tool_names": [t["name"] for t in tools]})
        if not self.script:
            raise LLMError("mock script exhausted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item(history) if callable(item) else item


# ---------------------------------------------------------------------------------------------- recording and replay


def recording_from_run(run: dict) -> dict:
    """The saved form of a LIVE run's model side. Refuses anything that is not an actual live run."""
    if run.get("mode") != "live":
        raise ValueError(f"only an actual LIVE run can be saved as a replay recording (this run's mode is {run.get('mode')!r})")
    if not run.get("model") or not run.get("model_turns"):
        raise ValueError("a live recording needs the model id and the model's turns")
    return {"schema": "agent_replay_v1", "mode": "live", "label": "RECORDED FROM AN ACTUAL LIVE RUN. ALL DATA IS SYNTHETIC.", "model": run["model"],
            "provider": run.get("provider"), "as_of": run["as_of"], "policy_hash": run["policy_hash"], "turns": run["model_turns"],
            "tool_output_sha256": {c["tool_call_id"]: c["output_sha256"] for c in run["tool_calls"] if "output_sha256" in c},
            "outcome": {"status": run["status"], "fallback_reason": run.get("fallback_reason")}}


class ReplayLLM:
    """Plays back recorded model turns in order. `source` is 'live_recording' for a file saved from a live run; unit tests use
    from_turns(..., source='test_fixture') so a fixture can never be mistaken for a recording of a live model."""
    mode = "replay"

    def __init__(self, recording: dict, source: str):
        self.recording, self.source = recording, source
        self.provider = recording.get("provider")  # the provider the RECORDING came from (None for older recordings and test fixtures)
        self.model_id = recording.get("model", "unknown")
        self._turns = list(recording["turns"])
        self.recorded_hashes = recording.get("tool_output_sha256", {})

    @classmethod
    def from_file(cls, path):
        rec = json.loads(Path(path).read_text(encoding="utf-8"))
        if rec.get("schema") != "agent_replay_v1" or rec.get("mode") != "live" or not rec.get("model"):
            raise ValueError("not a recording of a live run (needs schema agent_replay_v1, mode 'live' and a model id)")
        return cls(rec, "live_recording")

    @classmethod
    def from_turns(cls, turns: list, source: str = "test_fixture", model: str = "test-fixture"):
        return cls({"schema": "agent_replay_v1", "mode": source, "model": model, "turns": turns}, source)

    def generate(self, system, history, tools):
        if not self._turns:
            raise LLMError("replay exhausted: the recording has no further model turns")
        t = self._turns.pop(0)
        return LLMTurn(text=t.get("text"), tool_calls=[ToolCall(c["name"], dict(c["args"]), f"replay-{i}") for i, c in enumerate(t.get("tool_calls", []))])


# ---------------------------------------------------------------------------------------------- live (Gemini)


def _schema(types, d: dict):
    kinds = {"string": types.Type.STRING, "number": types.Type.NUMBER, "integer": types.Type.INTEGER, "boolean": types.Type.BOOLEAN,
             "object": types.Type.OBJECT, "array": types.Type.ARRAY}
    kw = {"type": kinds[d["type"]]}
    if d.get("description"):
        kw["description"] = d["description"]
    if d.get("enum"):
        kw["enum"] = list(d["enum"])
    if d["type"] == "object":
        kw["properties"] = {k: _schema(types, v) for k, v in d.get("properties", {}).items()}
        if d.get("required"):
            kw["required"] = list(d["required"])
    if d["type"] == "array":
        kw["items"] = _schema(types, d["items"])
    return types.Schema(**kw)


class GeminiClient:
    """Live client. Never prints or stores the key. All provider exceptions become LLMError with the key redacted.

    `mode` is 'live' ONLY when this class built the real SDK client itself. A client passed in (as the unit tests do with a fake transport) makes the
    mode 'mock', so a mocked transport can never be recorded or reported as a live run."""

    provider = "gemini"

    def __init__(self, model: str, client=None, temperature: float = 0.0, timeout_seconds: float = 60.0, limiter: RateLimiter = None):
        from google import genai
        from google.genai import types
        self._types = types
        self._key = os.environ.get(KEY_VAR)
        self.limiter = limiter or RateLimiter()  # pass the SAME limiter that paced the model-selection calls so the per-minute quota is shared
        self.timeout_seconds, self.timeout_ms = timeout_seconds, http_timeout_ms(timeout_seconds)
        if client is None:
            if not self._key:
                raise LLMError(f"{KEY_VAR} is not set: live Gemini is unavailable")
            client = genai.Client(api_key=self._key, http_options=types.HttpOptions(timeout=self.timeout_ms))
            self.mode = "live"
        else:
            self.mode = "mock"
        self.client, self.model_id, self.temperature = client, model, temperature

    def _redact(self, text: str) -> str:
        return text.replace(self._key, "<key>") if self._key else text

    def _declarations(self, tools):
        t = self._types
        return [t.Tool(function_declarations=[t.FunctionDeclaration(name=d["name"], description=d["description"], parameters=_schema(t, d["parameters"])) for d in tools])]

    def _contents(self, history):
        t, out = self._types, []
        for h in history:
            if h["role"] == "user":
                out.append(t.Content(role="user", parts=[t.Part(text=h["text"])]))
            elif h["role"] == "model":
                turn = h["turn"]
                if turn.raw is not None:
                    out.append(turn.raw)  # the provider's own content, unchanged
                else:
                    parts = [t.Part(text=turn.text)] if turn.text else [t.Part(function_call=t.FunctionCall(name=c.name, args=c.args, id=c.provider_id)) for c in turn.tool_calls]
                    out.append(t.Content(role="model", parts=parts))
            else:  # tool results go back as function responses; an optional notice (tool budget used up) rides in the SAME message, never as a second user message
                parts = [t.Part(function_response=t.FunctionResponse(id=r["call"].provider_id, name=r["call"].name, response=jsonable(r["payload"]))) for r in h["results"]]
                if h.get("notice"):
                    parts.append(t.Part(text=h["notice"]))
                out.append(t.Content(role="user", parts=parts))
        return out

    def generate(self, system, history, tools):
        t = self._types
        config = t.GenerateContentConfig(system_instruction=system, tools=self._declarations(tools), temperature=self.temperature,
                                         automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True),
                                         http_options=t.HttpOptions(timeout=self.timeout_ms))  # per-request timeout, in the SDK's unit (milliseconds)
        try:
            contents = self._contents(history)
            resp = self.limiter.call(lambda: self.client.models.generate_content(model=self.model_id, contents=contents, config=config))
            calls = [ToolCall(c.name, dict(c.args or {}), c.id) for c in (resp.function_calls or [])]
            text = None if calls else resp.text
            raw = resp.candidates[0].content if resp.candidates else None
            u = getattr(resp, "usage_metadata", None)
            usage = None if u is None else {"prompt_tokens": getattr(u, "prompt_token_count", None), "output_tokens": getattr(u, "candidates_token_count", None)}
        except Exception as exc:  # network, timeout, quota, safety block, malformed SDK response: all are API failures
            what = f"TIMEOUT after the configured {self.timeout_seconds:g}s ({self.timeout_ms} ms): " if isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower() else ""
            if getattr(exc, "rate_limit_kind", None) == "billing":
                what = "BILLING: THE PROJECT'S PREPAID CREDITS ARE DEPLETED (HTTP 402). This is an account state, not a rate limit; no retry was attempted. "
            elif getattr(exc, "rate_limit_kind", None) == "daily":
                ra = getattr(exc, "rate_limit_retry_after", None)
                what = ("DAILY QUOTA EXHAUSTED (HTTP 429): no retry was attempted" + (f"; the provider says it resets in about {ra / 3600:.1f} h" if ra else "") + ". ")
            elif getattr(exc, "rate_limit_retries", None) is not None:
                what = f"RATE LIMIT (HTTP 429) still in force after {exc.rate_limit_retries} bounded retr{'y' if exc.rate_limit_retries == 1 else 'ies'}: "
            raise LLMError(self._redact(f"{what}{type(exc).__name__}: {str(exc)[:600]}")) from None
        return LLMTurn(text=text, tool_calls=calls, raw=raw, usage=usage)


def list_generate_models(client) -> list:
    """Model ids the account can use for generateContent (as listed by the API)."""
    return [m.name for m in client.models.list() if "generateContent" in (m.supported_actions or [])]


def candidate_models(listing: list) -> list:
    """Order a LISTING (never a guess) for trial: text 'flash' models first, preview/lite/image/audio/embedding variants last. The caller must still
    verify each candidate with the live round-trip check before using it."""
    def rank(name):
        n = name.lower()
        bad = any(w in n for w in ("preview", "lite", "image", "tts", "audio", "live", "embed", "vision", "thinking", "exp", "robotics", "computer"))
        return (bad, "flash" not in n, "gemini" not in n, n)
    return sorted(listing, key=rank)
