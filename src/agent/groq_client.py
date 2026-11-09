"""Groq provider behind the existing LLM interface (added after the baseline freeze; NOT part of it). ALL DATA IS SYNTHETIC.

    client.generate(system, history, tools) -> LLMTurn          (same contract as MockLLM / ReplayLLM / GeminiClient)

What it does
  * Talks to Groq's OpenAI-compatible chat-completions endpoint with LOCAL function calling only: `tools` = our seven read-only functions as {"type":"function"}
    declarations. No hosted tool (no browser search, no code execution), no compound model. Provider tool-call ids are kept and every tool result is sent back
    with the id of the call it answers; a result that matches no call is refused before anything is sent.
  * The key is read ONLY from the GROQ_API_KEY environment variable, lives only inside the transport object, is never printed, stored or placed in a record, and is
    redacted from every error message.
  * Token-aware pacing from the account's limits (30 requests/min, 8000 tokens/min, 1000 requests/day, 200000 tokens/day for the candidate model): a request is only
    sent when its ESTIMATED tokens (prompt + the completion it may use) fit in what remains of the minute; the provider's rate-limit headers (when present) replace
    the local estimate; a request that cannot fit in the per-minute limit even alone is REFUSED before sending (waiting cannot fix an oversized request).
  * Bounded retry: a per-minute 429 waits for the provider's retry-after (capped) at most `max_retries` times; a DAILY limit, an oversized request (413), an
    authentication or billing error is never retried. Timeouts are not retried. All failures become GroqError(LLMError) so the agent takes its labelled rules-only fallback.
  * The model-visible tool results are a COMPACT view of the authoritative envelopes (no whitespace; output_sha256, args and the repeated as_of / policy_hash dropped
    after the first result). The authoritative envelopes, populations, the validator and every number it checks are untouched: this function only changes what is SENT.

`mode` is 'live' ONLY when this class built the real transport itself (reading the key). An injected transport (every unit test) makes the mode 'mock', so a faked
exchange can never be reported or recorded as a live run. `provider` is always 'groq'.

CHECKED BY ONE LIVE SMOKE TEST (reports/groq_smoke_status.json): the rate-limit header names (x-ratelimit-limit/remaining/reset for requests and tokens), the model id, the
reasoning_effort parameter, parallel-capable local function calling with `fc_...` call ids, and the id matching all worked. STAGED FINALIZATION (generate_structured, not yet used live): a second, tool-free request with response_format json_schema strict; see src/agent/strict_schema.py and agent.run_agent_staged.
THREE FULL CASES (reports/phase5_runs/live_groq_2025-11-10*.json) ended in the rules-only fallback after the provider rejected a model generation as a call to a
non-existent tool (HTTP 400 tool_use_failed: 'commentary', then the final answer as a call to 'JSON'; see unwrap_failed_generation). The headers show a continuously refilling token bucket and accounting close to usage (30-500 tokens above on some
responses). STILL UNVERIFIED: the 429/413 message wording the classifier reads, and whether the per-minute limit pre-reserves max_completion_tokens (assumed: yes).
"""
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass, field

from src.agent import strict_schema
from src.agent.llm import LLMError, LLMTurn, ToolCall
from src.ledger.proposal import jsonable

KEY_VAR = "GROQ_API_KEY"
ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
USER_AGENT = "player-value-agent/1.0 (research prototype; synthetic data)"

PROVIDER_NOTE = (
    "\n\nPROVIDER NOTE (this deployment has a small tokens-per-minute limit, so be economical). Start with get_review_overview. Then request get_policy_decision for ALL paid "
    "channels TOGETHER in ONE turn as parallel tool calls (they are independent; each separate turn re-sends the whole history and costs tokens). Call get_quality_checks, get_economics, get_forecast_and_populations or plan_experiment ONLY for a channel where the evidence looks "
    "incomplete, a stage failed, the policy proposes a decrease or an increase, or something does not add up; do not sweep every tool for every channel. Batch any other independent calls the same way. Tool results are shown as compact JSON: after the first result the repeated fields as_of, policy_hash, args and output_sha256 are omitted (they are "
    "unchanged and are checked by the validator); the result and its population labels are complete. Write the final JSON as ordinary message text, never as a tool call (there is no tool named JSON). Keep the final JSON compact.")


STAGED_INVESTIGATION_NOTE = (
    "\n\nSTAGED PROTOCOL. Do NOT write the final answer in this conversation. Investigate with the tools; when you have the evidence you need (including get_policy_decision for every paid "
    "channel), reply with the single word DONE. A separate structured request, in which no tools exist, will then ask for the proposal.")

FINALIZATION_NOTE = (
    "\n\nSTRUCTURED FINALIZATION. The investigation is finished and NO tools are available now. Using ONLY the evidence index in the user message, write the final proposal as the JSON "
    "object the response schema requires. Cite each claim with the call id, field path, value and population label exactly as the index lists them. Every paid channel needs one item that "
    "cites its get_policy_decision action. Do not invent a call, a field, a value or a label; the schema fixes only the SHAPE of your answer, and every claim is still checked against the evidence.")


class GroqError(LLMError):
    """An API failure. `kind` is one of: daily | per_minute | request_too_large | oversized_preflight | auth | billing | timeout | network | server | tool_use_failed | schema_failure | refusal | incomplete_output | bad_request | malformed_response."""

    def __init__(self, message, kind="api"):
        super().__init__(message)
        self.kind = kind


# ---------------------------------------------------------------------------------------------- durations and error classification

_DUR = re.compile(r"(?:(\d+(?:\.\d+)?)d)?(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m(?!s))?(?:(\d+(?:\.\d+)?)s)?(?:(\d+(?:\.\d+)?)ms)?")


def parse_duration(text):
    """Seconds in '7.66s', '2m59.56s', '1h2m', '340ms', '1d2h' or a bare number of seconds; None if `text` is not such a value."""
    t = str(text).strip().replace(" ", "")
    if not t:
        return None
    try:
        return float(t)
    except ValueError:
        pass
    m = _DUR.fullmatch(t)
    if not m or not any(m.groups()):
        return None
    d, h, mi, s, ms = (float(g) if g else 0.0 for g in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + s + ms / 1000.0


@dataclass
class HttpResult:
    status: int
    headers: dict  # lower-cased names
    text: str


@dataclass
class ErrorInfo:
    kind: str
    retry_after: float = None
    message: str = ""


def _error_message(text: str) -> str:
    try:
        e = json.loads(text).get("error", {})
        return str(e.get("message") or text)[:600] if isinstance(e, dict) else text[:600]
    except (ValueError, AttributeError):
        return str(text)[:600]


def classify_error(res: HttpResult) -> ErrorInfo:
    """What kind of failure an error response is. Daily limits and request-size errors are recognised from the message wording (token/request 'per day' or 'per minute',
    'request too large'); the retry delay comes from the retry-after header, else the message's 'try again in <duration>'."""
    msg = _error_message(res.text)
    low = msg.lower()
    retry = parse_duration(res.headers.get("retry-after", "")) if res.headers.get("retry-after") else None
    if retry is None:
        m = re.search(r"try again in\s+([0-9dhms.]+)", msg, re.IGNORECASE)
        retry = parse_duration(m.group(1).rstrip(".")) if m else None
    if res.status in (401, 403):
        return ErrorInfo("auth", None, msg)
    if res.status == 402:
        return ErrorInfo("billing", None, msg)
    if res.status == 413 or "request too large" in low:
        return ErrorInfo("request_too_large", None, msg)
    if res.status == 400 and re.search(r"json_validate_failed|failed to generate json", low + " " + res.text.lower()):
        return ErrorInfo("schema_failure", None, msg)  # structured output: the provider could not produce JSON that satisfies the schema
    if res.status == 400 and re.search(r"tool_use_failed|tool call validation failed", low + " " + res.text.lower()):
        return ErrorInfo("tool_use_failed", None, msg)  # the MODEL produced a call the provider could not validate (seen live: a call to a non-existent tool 'commentary')
    if res.status == 429:
        if re.search(r"per\s*day|\bdaily\b|\(tpd\)|\(rpd\)", low):
            return ErrorInfo("daily", retry, msg)
        return ErrorInfo("per_minute", retry, msg)
    if res.status >= 500:
        return ErrorInfo("server", retry, msg)
    return ErrorInfo("bad_request", None, msg)


# ---------------------------------------------------------------------------------------------- pacing


class TokenPacer:
    """Token-BUCKET pacing, fitted to what the provider's headers showed in two live exchanges: capacity = the per-minute token limit, refilled continuously at limit/60 per
    second (every recorded `x-ratelimit-reset-tokens` equalled tokens-used / (8000/60) exactly, i.e. the time for the bucket to refill). A request is admitted when the tokens
    that have refilled cover its ESTIMATE (prompt + the completion it may use); otherwise we sleep just long enough. The provider's own `remaining-tokens` header, when present,
    REPLACES our level after each response. A request larger than the capacity (minus a margin) can never be admitted and is refused. A sliding 60-second window per-request
    count enforces requests per minute. Pure time logic: clock and sleep are injectable so tests need no real waiting."""

    def __init__(self, tpm=8000, rpm=30, margin_tokens=300, window=60.0, max_wait=90.0, clock=time.monotonic, sleep=time.sleep):
        self.tpm, self.rpm, self.margin, self.window, self.max_wait, self._clock, self._sleep = tpm, rpm, margin_tokens, window, max_wait, clock, sleep
        self.rate = tpm / window  # tokens refilled per second
        self.level, self.level_t = float(tpm), clock()  # the bucket starts full: the first headers correct it if the account was already in use
        self.starts = deque()  # request start times inside the last window
        self.server = {}  # last provider-reported numbers
        self.waited = 0.0
        self.tokens_used_total = 0
        self.requests_made = 0

    @property
    def cap(self) -> int:
        return self.tpm - self.margin

    def available(self, now=None) -> float:
        now = self._clock() if now is None else now
        return min(float(self.tpm), self.level + max(0.0, now - self.level_t) * self.rate)

    def wait_needed(self, est: int) -> float:
        now = self._clock()
        while self.starts and now - self.starts[0] >= self.window:
            self.starts.popleft()
        need = 0.0
        if len(self.starts) >= self.rpm:
            need = self.starts[len(self.starts) - self.rpm] + self.window - now
        short = est - self.available(now)
        if short > 0:
            need = max(need, short / self.rate)
        return max(0.0, need)

    def acquire(self, est: int) -> list:
        if est > self.cap:
            raise GroqError(f"REQUEST TOO LARGE (preflight): this request is estimated at {est} tokens but only {self.cap} fit in one minute "
                            f"(limit {self.tpm} tokens/min minus a {self.margin}-token margin). Waiting cannot fix an oversized request, so it was NOT sent.", "oversized_preflight")
        need = self.wait_needed(est)
        if need > self.max_wait:
            raise GroqError(f"PACING: the next request would have to wait about {need:.0f}s (more than the allowed {self.max_wait:g}s); it was not sent.", "per_minute")
        if need > 0:
            self._sleep(need + 0.05)
            self.waited += need + 0.05
        now = self._clock()
        self.level, self.level_t = self.available(now) - est, now  # reserve the estimate
        self.starts.append(now)
        self.requests_made += 1
        return [now, est]

    def record(self, ev: list, actual_total: int = None, headers: dict = None):
        """A response arrived. Replace the reserved estimate by the real usage (or, when the provider sent its remaining-tokens figure, by that figure)."""
        now = self._clock()
        if actual_total is not None:
            self.level += ev[1] - actual_total  # give back the over-reservation (the level may go negative if usage exceeded the estimate)
            ev[1] = actual_total
            self.tokens_used_total += actual_total
        self._from_headers(headers or {}, now)

    def _from_headers(self, h: dict, now: float) -> bool:
        rt, rr = h.get("x-ratelimit-remaining-tokens"), h.get("x-ratelimit-remaining-requests")
        ok = False
        if rt is not None:
            try:
                self.level, self.level_t = float(rt), now  # authoritative: the provider's bucket at the moment it answered
                self.server["remaining_tokens"] = int(float(rt))
                d = parse_duration(h.get("x-ratelimit-reset-tokens", ""))
                self.server["tokens_reset_at"] = None if d is None else now + d
                ok = True
            except ValueError:
                pass
        if rr is not None:
            try:
                self.server["remaining_requests_day"] = int(float(rr))
            except ValueError:
                pass
        return ok

    def rejected(self, ev: list, headers: dict = None):
        """An error response. If the provider reported its bucket (even a failed generation can consume tokens), trust that; otherwise the request used none: give the reservation back."""
        now = self._clock()
        if not self._from_headers(headers or {}, now):
            self.level = min(float(self.tpm), self.level + ev[1])
            self.requests_made -= 1
            try:
                self.starts.remove(ev[0])
            except ValueError:
                pass


# ---------------------------------------------------------------------------------------------- transport


class UrllibTransport:
    """The ONLY object that holds the key. Standard library only (no new dependency). Returns every HTTP status as an HttpResult; raises TimeoutError / OSError."""

    def __init__(self, api_key: str, url: str = ENDPOINT):
        self._key, self.url = api_key, url

    def __repr__(self):
        return f"UrllibTransport(url={self.url!r}, key=<hidden>)"

    def __call__(self, body: dict, timeout_seconds: float) -> HttpResult:
        req = urllib.request.Request(self.url, data=json.dumps(body, separators=(",", ":")).encode("utf-8"), method="POST",
                                     headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json", "Accept": "application/json", "User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as r:  # urllib's timeout is in SECONDS (the Gemini SDK's is milliseconds)
                return HttpResult(r.status, {k.lower(): v for k, v in r.headers.items()}, r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            return HttpResult(e.code, {k.lower(): v for k, v in e.headers.items()}, e.read().decode("utf-8", "replace"))
        except urllib.error.URLError as e:
            if isinstance(e.reason, TimeoutError):
                raise e.reason from None
            raise


# ---------------------------------------------------------------------------------------------- the client


def model_view(payload: dict, first: bool) -> str:
    """The text the model SEES for one tool result: compact JSON of the authoritative envelope without output_sha256 and args, and without as_of / policy_hash after the first
    result. tool_call_id, tool, channel, populations, result (or error) are kept exactly. The envelope held by the runner and checked by the validator is NOT modified."""
    drop = {"output_sha256", "args"} | (set() if first else {"as_of", "policy_hash"})
    return json.dumps({k: v for k, v in jsonable(payload).items() if k not in drop}, separators=(",", ":"), ensure_ascii=False)


def unwrap_failed_generation(res_text: str, tool_names) -> str:
    """When the provider rejects a generation as 'a call to a tool that is not in request.tools', its error body carries the model's raw text in `failed_generation`.
    Observed live: the model's FINAL ANSWER arrived as {"name": "JSON", "arguments": {...answer...}}. If (and only if) that text parses as a call to a name that is NOT one of our tools whose
    arguments are an object with the answer's two top-level keys (`summary`, `items`), return the arguments as JSON text, unchanged; otherwise None. Nothing is repaired, completed or
    reinterpreted: the validator then checks the text like any other answer, and the run is labelled."""
    try:
        fg = json.loads(res_text)["error"]["failed_generation"]
        obj = json.loads(fg) if isinstance(fg, str) else None
    except (ValueError, KeyError, TypeError):
        return None
    if not isinstance(obj, dict) or "name" not in obj or "arguments" not in obj or obj["name"] in set(tool_names):
        return None
    args = obj["arguments"]
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return None
    if isinstance(args, dict) and "summary" in args and "items" in args:
        return json.dumps(args, separators=(",", ":"), ensure_ascii=False)
    return None


class GroqClient:
    provider = "groq"

    def __init__(self, model: str, cfg: dict = None, transport=None, clock=time.monotonic, sleep=time.sleep):
        cfg = cfg or {}
        self.model_id, self.cfg = model, cfg
        self.temperature = cfg.get("temperature", 0.0)
        self.timeout_seconds = float(cfg.get("timeout_seconds", 60))
        if not 0 < self.timeout_seconds < float("inf"):
            raise ValueError(f"timeout_seconds must be a positive number, got {self.timeout_seconds!r}")
        self.max_completion_tokens = int(cfg.get("max_completion_tokens", 2600))
        self.min_completion_tokens = int(cfg.get("min_completion_tokens", 700))
        self.reasoning_effort = cfg.get("reasoning_effort", "low")
        self.max_retries, self.max_wait = int(cfg.get("max_rate_limit_retries", 2)), float(cfg.get("max_rate_limit_wait_seconds", 90))
        self.max_tool_use_retries = int(cfg.get("max_tool_use_failed_retries", 0))
        self.unwrap_wrapped_answers = bool(cfg.get("unwrap_wrapped_final_answer", True))  # single-stage runs only: a staged run writes no answer during investigation
        self.staged = bool(cfg.get("staged_finalization", False))  # investigation with tools, then a separate tool-free request constrained by a strict JSON schema
        self.max_evidence_nudges = int(cfg.get("max_evidence_nudges", 2))
        self.chars_per_token = float(cfg.get("assumed_chars_per_token", 2.8))  # conservative until a response shows the real figure
        self._sleep = sleep
        self.pacer = TokenPacer(cfg.get("tpm", 8000), cfg.get("rpm", 30), cfg.get("margin_tokens", 300), 60.0, self.max_wait, clock, sleep)
        self.request_log = []
        self.salvaged = 0  # model answers unwrapped from a provider tool_use_failed error (see unwrap_failed_generation)
        self._key = os.environ.get(KEY_VAR)
        if transport is None:
            if not self._key:
                raise GroqError(f"{KEY_VAR} is not set: live Groq is unavailable", "auth")
            transport, self.mode = UrllibTransport(self._key), "live"
        else:
            self.mode = "mock"  # an injected transport can never be reported as a live run
        self.transport = transport

    def run_notes(self) -> list:
        """Labels the agent loop copies into the proposal's labels: anything that happened at the provider that a reviewer must know."""
        if not self.salvaged:
            return []
        return [f"PROVIDER FORMAT RECOVERY: {self.salvaged} model answer(s) were rejected by the provider as a call to a tool that does not exist (HTTP 400 tool_use_failed) and were unwrapped from its "
                "failed_generation field unchanged. The validator checked the text like any other answer."]

    # ---- serialisation
    def investigation_note(self) -> str:
        return PROVIDER_NOTE + (STAGED_INVESTIGATION_NOTE if self.staged else "")

    def _redact(self, text: str) -> str:
        return text.replace(self._key, "<key>") if self._key else text

    def messages(self, system, history, note=None) -> list:
        msgs, ids, n_ids, first_result = [{"role": "system", "content": system + (self.investigation_note() if note is None else note)}], {}, 0, True
        pending = None  # ids of the last assistant tool calls still waiting for results
        for h in history:
            if h["role"] == "user":
                if pending:
                    raise GroqError("MALFORMED HISTORY: a user message follows tool calls that have no tool result", "bad_request")
                msgs.append({"role": "user", "content": h["text"]})
            elif h["role"] == "model":
                if pending:
                    raise GroqError("MALFORMED HISTORY: a model turn follows tool calls that have no tool result", "bad_request")
                t = h["turn"]
                if t.tool_calls:
                    tcs = []
                    for c in t.tool_calls:
                        n_ids += 1
                        cid = c.provider_id or f"call_{n_ids:03d}"
                        if cid in ids.values():
                            cid = f"{cid}_{n_ids}"
                        ids[id(c)] = cid
                        tcs.append({"id": cid, "type": "function", "function": {"name": c.name, "arguments": json.dumps(jsonable(c.args), separators=(",", ":"))}})
                    msgs.append({"role": "assistant", "content": None, "tool_calls": tcs})
                    pending = [x["id"] for x in tcs]
                else:
                    msgs.append({"role": "assistant", "content": t.text or ""})
            else:  # tool results: one tool message per call, in the order of the calls, each with the id of ITS call
                got = []
                for r in h["results"]:
                    cid = ids.get(id(r["call"]))
                    if cid is None:
                        raise GroqError("MALFORMED HISTORY: a tool result matches no assistant tool call", "bad_request")
                    got.append(cid)
                    msgs.append({"role": "tool", "tool_call_id": cid, "content": model_view(r["payload"], first_result)})
                    first_result = False
                if sorted(got) != sorted(pending or []):
                    raise GroqError(f"MALFORMED HISTORY: tool results {sorted(got)} do not match the assistant's tool calls {sorted(pending or [])}", "bad_request")
                pending = None
                if h.get("notice"):
                    msgs.append({"role": "user", "content": h["notice"]})
        return msgs

    @staticmethod
    def tool_declarations(tools) -> list:
        return [{"type": "function", "function": {"name": d["name"], "description": d["description"], "parameters": d["parameters"]}} for d in tools]

    @staticmethod
    def _chars(body: dict) -> int:
        """Characters the provider is charged for, as one serialisation: messages plus tool declarations or the response schema."""
        return len(json.dumps({"messages": body["messages"], "tools": body.get("tools", []), "response_format": body.get("response_format")}, separators=(",", ":"), ensure_ascii=False))

    def estimate_tokens(self, body: dict) -> int:
        return int(self._chars(body) / self.chars_per_token) + 1

    def build_body(self, system, history, tools) -> dict:
        body = {"model": self.model_id, "messages": self.messages(system, history), "tools": self.tool_declarations(tools), "tool_choice": "auto", "parallel_tool_calls": True,
                "temperature": self.temperature, "max_completion_tokens": self.max_completion_tokens, "stream": False}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        return body

    def build_structured_body(self, system, history, schema=None) -> dict:
        """The FINALIZATION request: no tool declarations, no tool_choice, no parallel_tool_calls, not streamed; a strict json_schema response_format derived from the output contract."""
        body = {"model": self.model_id, "messages": self.messages(system, history, note=FINALIZATION_NOTE), "response_format": strict_schema.response_format(schema),
                "temperature": self.temperature, "max_completion_tokens": self.max_completion_tokens, "stream": False}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        return body

    # ---- one request
    def generate(self, system, history, tools):
        return self._run(self.build_body(system, history, tools), structured=False)

    def generate_structured(self, system, history, schema=None):
        """Stage 2: ONE non-streaming request without tools whose output must satisfy the strict schema. Returns an LLMTurn whose text is the JSON string."""
        return self._run(self.build_structured_body(system, history, schema), structured=True)

    def _run(self, body, structured):
        prompt_est = self.estimate_tokens(body)
        # OUTPUT HEADROOM: the prompt plus the completion it may use must fit in one minute's tokens; shrink the completion cap to fit, refuse if too little is left
        room = self.pacer.cap - prompt_est
        body["max_completion_tokens"] = min(self.max_completion_tokens, room)
        if body["max_completion_tokens"] < self.min_completion_tokens:
            self.request_log.append({"prompt_tokens_estimated": prompt_est, "max_completion_tokens": None, "attempts": 0, "refused_before_sending": True,
                                     "n_messages": len(body["messages"]), "n_tool_messages": sum(1 for m in body["messages"] if m["role"] == "tool")})
            raise GroqError(f"REQUEST TOO LARGE (preflight): the prompt is estimated at {prompt_est} tokens, leaving {max(room, 0)} for the answer inside the {self.pacer.cap}-token minute "
                            f"(at least {self.min_completion_tokens} are needed). Waiting cannot fix this, so the request was NOT sent.", "oversized_preflight")
        est_total = prompt_est + body["max_completion_tokens"]
        rec = {"prompt_tokens_estimated": prompt_est, "max_completion_tokens": body["max_completion_tokens"], "attempts": 0, "slept_seconds": 0.0,
               "stage": "finalization" if structured else "investigation" if self.staged else "single_stage",
               "provider_note_sha256": hashlib.sha256((FINALIZATION_NOTE if structured else self.investigation_note()).encode()).hexdigest()[:16]}
        if structured:
            rec["schema_sha256"] = hashlib.sha256(json.dumps(body["response_format"], sort_keys=True).encode()).hexdigest()[:16]
        self.request_log.append(rec)
        attempt = 0
        while True:
            before = self.pacer.waited
            ev = self.pacer.acquire(est_total)
            rec["slept_seconds"] = round(rec["slept_seconds"] + self.pacer.waited - before, 2)
            rec["attempts"] += 1
            try:
                res = self.transport(body, self.timeout_seconds)
            except TimeoutError:
                self.pacer.record(ev)
                raise GroqError(f"TIMEOUT after the configured {self.timeout_seconds:g}s: no answer was received (not retried)", "timeout") from None
            except OSError as e:
                self.pacer.rejected(ev)
                raise GroqError(self._redact(f"NETWORK ERROR: {type(e).__name__}: {str(e)[:300]}"), "network") from None
            rec["status"] = res.status
            rec["headers"] = {k: v for k, v in res.headers.items() if k.startswith("x-ratelimit") or k == "retry-after"}
            if res.status == 200:
                return self._parse_structured(res, ev, rec, body) if structured else self._parse(res, ev, rec, body)
            self.pacer.rejected(ev, res.headers)  # the provider's own bucket figure, if it sent one (a failed generation can still consume tokens)
            info = classify_error(res)
            rec["error_kind"] = info.kind
            rec["error_excerpt"] = self._redact(res.text[:6000])  # what the provider said (model text only, never a key), kept for diagnosis
            if info.kind == "tool_use_failed" and self.unwrap_wrapped_answers and not self.staged and not structured:
                text = unwrap_failed_generation(res.text, [t["function"]["name"] for t in body["tools"]])
                if text is not None:
                    self.salvaged += 1
                    rec["unwrapped_failed_generation"] = True
                    return LLMTurn(text=text, tool_calls=[], raw=None, usage={"prompt_tokens": None, "output_tokens": None, "finish_reason": "unwrapped_from_tool_use_failed"})
            retryable = info.kind in ("per_minute", "server", "tool_use_failed")
            delay = info.retry_after if info.retry_after is not None else (2.0 * (attempt + 1) if info.kind == "server" else 1.0 if info.kind == "tool_use_failed" else 20.0)
            if info.kind == "per_minute" and info.retry_after is None:
                d = parse_duration(res.headers.get("x-ratelimit-reset-tokens", ""))
                delay = d if d is not None else delay
            limit = self.max_tool_use_retries if info.kind == "tool_use_failed" else self.max_retries
            if not retryable or attempt >= limit or delay > self.max_wait:
                raise GroqError(self._redact(self._label(info, res.status, attempt, delay)), info.kind)
            attempt += 1
            self._sleep(delay + 0.5)
            self.pacer.waited += delay + 0.5
            rec["slept_seconds"] = round(rec["slept_seconds"] + delay + 0.5, 2)

    @staticmethod
    def _label(info: ErrorInfo, status: int, attempt: int, delay: float) -> str:
        what = {"daily": "DAILY LIMIT REACHED (HTTP 429): no retry was attempted. ", "request_too_large": "REQUEST TOO LARGE (HTTP 413): waiting cannot fix an oversized request; no retry. ",
                "auth": "AUTHENTICATION FAILED: the key was rejected (not retried). ",
                "schema_failure": "STRUCTURED OUTPUT FAILED (HTTP 400): the provider could not produce JSON matching the schema; not retried. ",
                "tool_use_failed": f"MODEL TOOL-CALL FORMAT FAILURE (HTTP 400 tool_use_failed) still present after {attempt} bounded retr{'y' if attempt == 1 else 'ies'}: the provider could not validate a call the model produced. ", "billing": "BILLING / PLAN ERROR (HTTP 402): not retried. ",
                "per_minute": f"RATE LIMIT (HTTP 429) still in force after {attempt} bounded retr{'y' if attempt == 1 else 'ies'} (the provider asked for {delay:g}s). ",
                "server": f"PROVIDER SERVER ERROR (HTTP {status}) after {attempt} bounded retr{'y' if attempt == 1 else 'ies'}. ", "bad_request": f"REQUEST REJECTED (HTTP {status}): "}[info.kind]
        return f"{what}{info.message}"

    def _parse(self, res: HttpResult, ev: list, rec: dict, body: dict) -> LLMTurn:
        try:
            data = json.loads(res.text)
            choice = data["choices"][0]
            msg = choice["message"]
            if not isinstance(msg, dict):
                raise TypeError("message is not an object")
        except (ValueError, KeyError, IndexError, TypeError) as e:
            self.pacer.record(ev, None, res.headers)
            raise GroqError(f"MALFORMED RESPONSE: no usable choices[0].message ({type(e).__name__})", "malformed_response") from None
        u = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        total = u.get("total_tokens") or ((u.get("prompt_tokens") or 0) + (u.get("completion_tokens") or 0)) or None
        self.pacer.record(ev, total, res.headers)
        if u.get("prompt_tokens"):  # calibrate: the next estimate uses the figure THIS provider actually charged for the same serialisation
            chars = self._chars(body)
            self.chars_per_token = max(1.5, min(6.0, chars / u["prompt_tokens"] * 0.95))
            rec["observed_chars_per_token"] = round(chars / u["prompt_tokens"], 3)
        rec["usage"] = {k: u.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
        rec["finish_reason"] = choice.get("finish_reason")
        calls = []
        raw_calls = msg.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            raise GroqError("MALFORMED RESPONSE: tool_calls is not a list", "malformed_response")
        for c in raw_calls:
            try:
                fn = c["function"]
                name = fn["name"]
                if not isinstance(name, str) or not name:
                    raise TypeError("function name missing")
            except (KeyError, TypeError):
                raise GroqError("MALFORMED RESPONSE: a tool call has no function name", "malformed_response") from None
            try:
                args = json.loads(fn.get("arguments") or "{}")
                if not isinstance(args, dict):
                    raise ValueError("arguments are not an object")
            except ValueError:
                # not silently repaired: the tool runner rejects it (bad_arguments), the model sees the error, and the call counts against the budget
                args = {"_unparsed_arguments": str(fn.get("arguments"))[:200]}
            calls.append(ToolCall(name, args, c.get("id") if isinstance(c.get("id"), str) else None))
        text = None if calls else (msg.get("content") if isinstance(msg.get("content"), str) else "")
        usage = {"prompt_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens"), "finish_reason": choice.get("finish_reason")}
        return LLMTurn(text=text, tool_calls=calls, raw=None, usage=usage)

    def _parse_structured(self, res: HttpResult, ev: list, rec: dict, body: dict) -> LLMTurn:
        """A 200 response to the finalization request: explicit handling of a refusal, incomplete output and a missing or non-text answer. Whether the text is valid JSON that the
        validator accepts is the AGENT's business (the strict schema makes valid JSON the expected case, never a proof of a correct answer)."""
        try:
            data = json.loads(res.text)
            choice = data["choices"][0]
            msg = choice["message"]
            if not isinstance(msg, dict):
                raise TypeError("message is not an object")
        except (ValueError, KeyError, IndexError, TypeError) as e:
            self.pacer.record(ev, None, res.headers)
            raise GroqError(f"MALFORMED RESPONSE: no usable choices[0].message ({type(e).__name__})", "malformed_response") from None
        u = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        total = u.get("total_tokens") or ((u.get("prompt_tokens") or 0) + (u.get("completion_tokens") or 0)) or None
        self.pacer.record(ev, total, res.headers)
        if u.get("prompt_tokens"):
            chars = self._chars(body)
            self.chars_per_token = max(1.5, min(6.0, chars / u["prompt_tokens"] * 0.95))
            rec["observed_chars_per_token"] = round(chars / u["prompt_tokens"], 3)
        rec["usage"] = {k: u.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
        finish = choice.get("finish_reason")
        rec["finish_reason"] = finish
        if msg.get("refusal"):
            rec["refusal"] = str(msg["refusal"])[:300]
            raise GroqError(self._redact(f"MODEL REFUSED the structured request: {str(msg['refusal'])[:300]}"), "refusal")
        if finish == "length":
            raise GroqError(f"OUTPUT INCOMPLETE (finish_reason=length): the structured answer reached max_completion_tokens={body['max_completion_tokens']} and was cut off; it was not used.", "incomplete_output")
        if msg.get("tool_calls"):
            raise GroqError("MALFORMED RESPONSE: a tool call came back from a request that declared no tools", "malformed_response")
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            raise GroqError(f"MALFORMED RESPONSE: the structured request returned no text content (finish_reason={finish})", "malformed_response")
        try:
            json.loads(content)
            rec["content_is_json"] = True
        except ValueError:
            rec["content_is_json"] = False  # recorded: the strict schema should make this impossible; the agent treats it as malformed output
        return LLMTurn(text=content, tool_calls=[], raw=None, usage={"prompt_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens"), "finish_reason": finish})
