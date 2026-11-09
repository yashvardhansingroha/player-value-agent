"""Small Groq smoke test (Groq provider). ALL DATA IS SYNTHETIC. Result of a run is a CONNECTIVITY / CALIBRATION check, NOT agent evidence.

    python scripts/groq_smoke.py            # OFFLINE (default): prints the planned requests and token estimates. No key needed, NO API call.
    python scripts/groq_smoke.py --live     # TWO real requests, only with an explicit --live and GROQ_API_KEY in the environment

What --live does (about 5,000 to 6,500 tokens in total, inside one minute's 8,000):
  request 1  the real system prompt + provider note + all seven tool declarations + the real task prompt for the 2025-10-13 dev review. The model should call tools.
             We execute them with the real read-only tools on the DEV seed, and send the results back.
  request 2  the tool results (compact view) plus a smoke-test notice asking for ONE short sentence and no tools.
It records, per request: estimated vs actual prompt tokens (so the token estimate is calibrated), the provider's rate-limit headers, finish reason, the response fields it
returned, and any error with its label. Nothing here is a validated proposal and nothing is saved as a replay recording. The key is read only from the environment and is
never printed or written. Output: reports/groq_smoke_status.json.
"""
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_agent import build_environment  # noqa: E402  (harness: DEV seed 1 only)
from src.agent import agent as A  # noqa: E402
from src.agent import tools as T  # noqa: E402
from src.agent.groq_client import KEY_VAR, GroqClient, GroqError  # noqa: E402
from src.agent.llm import LLMError  # noqa: E402

OUT = ROOT / "reports" / "groq_smoke_status.json"
AS_OF = "2025-10-13"
NOTICE = "SMOKE TEST ONLY: do not call any more tools. Reply with ONE short sentence saying how many tool results you received. This is not a budget review answer."


def main(live: bool) -> int:
    cfg = yaml.safe_load((ROOT / "config" / "agent_groq.yaml").read_text(encoding="utf-8"))
    ctx = build_environment(AS_OF)
    task = [{"role": "user", "text": A.task_prompt(ctx, cfg["max_tool_calls"])}]
    status = {"label": "GROQ SMOKE TEST (connectivity and token calibration; NOT agent evidence)", "provider": "groq", "model": cfg["model"], "as_of": AS_OF, "live": live, "steps": []}
    if not live:
        c = GroqClient(cfg["model"], cfg, transport=lambda body, t: None)  # injected: no key read, no call possible
        body = c.build_body(A.SYSTEM_PROMPT, task, T.declarations())
        est = c.estimate_tokens(body)
        print(f"OFFLINE. Request 1 would be about {est} estimated tokens (assumed {cfg['assumed_chars_per_token']} chars/token) with a completion cap of "
              f"{min(cfg['max_completion_tokens'], c.pacer.cap - est)}; the per-minute budget is {cfg['tpm']} (margin {cfg['margin_tokens']}). No API call was made.")
        print("To run the live smoke test: set GROQ_API_KEY in your own shell, then add --live.")
        return 0
    if not os.environ.get(KEY_VAR):
        print(f"PENDING: {KEY_VAR} is not set; nothing was called.")
        return 0
    client = GroqClient(cfg["model"], cfg)  # reads the key itself -> mode 'live'
    status["mode"] = client.mode
    runner = T.ToolRunner(ctx, cfg["max_tool_calls"])
    history = list(task)
    try:
        turn = client.generate(A.SYSTEM_PROMPT, history, T.declarations())
        status["steps"].append({"step": "tool_call_request", "tool_calls": [{"name": c.name, "args": c.args, "provider_id": c.provider_id} for c in turn.tool_calls], "text": turn.text, "usage": turn.usage})
        if not turn.tool_calls:
            status["result"] = "The model answered without calling a tool; the round trip was not exercised."
        else:
            history.append({"role": "model", "turn": turn})
            results = [{"call": c, "payload": runner.deny(c) if runner.exhausted else runner.execute(c)} for c in turn.tool_calls]
            history.append({"role": "tool", "results": results, "notice": NOTICE})
            turn2 = client.generate(A.SYSTEM_PROMPT, history, T.declarations())
            status["steps"].append({"step": "round_trip_request", "n_results_sent": len(results), "text": turn2.text, "tool_calls_requested": len(turn2.tool_calls), "usage": turn2.usage})
            status["result"] = "round trip completed" if turn2.text else "the second reply had no text"
    except LLMError as e:
        status["result"] = f"FAILED: {e}"
        status["error_kind"] = getattr(e, "kind", None)
    status["request_log"] = client.request_log
    status["pacer"] = {"tokens_used_total": client.pacer.tokens_used_total, "requests_made": client.pacer.requests_made, "slept_seconds": round(client.pacer.waited, 1), "provider_numbers": client.pacer.server}
    ratios = [r["observed_chars_per_token"] for r in client.request_log if "observed_chars_per_token" in r]
    if ratios:
        status["calibration"] = {"observed_chars_per_token": ratios, "note": "compare with the offline model's 2.6 / 3.2 / 3.8 (pessimistic / central / optimistic) before any full run"}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(status, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: status[k] for k in ("result", "calibration", "pacer") if k in status}, indent=1, default=str))
    return 0 if str(status.get("result", "")).startswith("round trip") else 1


if __name__ == "__main__":
    sys.exit(main("--live" in sys.argv[1:]))
