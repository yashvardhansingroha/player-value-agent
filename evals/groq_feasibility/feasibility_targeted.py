import json, sys
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.run_agent import build_environment
from src.agent import agent as A
from src.agent.groq_client import GroqClient, HttpResult, GroqError
from src.agent.llm import LLMTurn, ToolCall
CFG = yaml.safe_load((ROOT / "config" / "agent_groq.yaml").read_text(encoding="utf-8"))

class Clock:
    t = 0.0
    def now(self): return self.t
    def sleep(self, s): self.t += s

def targeted_turns(ctx):
    chans = ctx.channels
    t1 = [("get_review_overview", {})]
    t2 = [("get_policy_decision", {"channel": c}) for c in chans]
    t3 = []
    for c in chans:
        a = ctx.decision(c)["action"]
        if a in ("insufficient_evidence", "decrease"):
            t3.append(("get_quality_checks", {"channel": c}))
        if a == "decrease":
            t3.append(("get_economics", {"channel": c}))
        if a == "increase_test":
            t3.append(("plan_experiment", {"channel": c, "new_budget": round(ctx.decision(c)["proposed_budget"], 6)}))
    return [t for t in (t1, t2, t3) if t]

def run(as_of, ratio, answer_tokens=1400, sequential=False, max_turns=None):
    ctx = build_environment(as_of)
    batches = targeted_turns(ctx)
    if sequential:  # what the live model did: ONE tool call per turn
        batches = [[c] for b in batches for c in b]
    ck, log = Clock(), []
    def transport(body, timeout):
        n = len(log)
        chars = len(json.dumps({"messages": body["messages"], "tools": body["tools"]}, separators=(",", ":"), ensure_ascii=False))
        prompt = int(chars / ratio)
        if n < len(batches):
            msg = {"content": None, "tool_calls": [{"id": f"call_{n}_{i}", "type": "function", "function": {"name": nm, "arguments": json.dumps(a)}} for i, (nm, a) in enumerate(batches[n])]}
            out = 200 + 30 * len(batches[n])
        else:
            msg, out = {"content": "{}"}, answer_tokens
        log.append({"t": round(ck.t, 1), "prompt_tokens": prompt, "completion_tokens": out, "max_completion_tokens": body["max_completion_tokens"]})
        return HttpResult(200, {}, json.dumps({"choices": [{"message": msg, "finish_reason": "stop"}], "usage": {"prompt_tokens": prompt, "completion_tokens": out, "total_tokens": prompt + out}}))
    c = GroqClient(CFG["model"], CFG, transport=transport, clock=ck.now, sleep=ck.sleep)
    try:
        r = A.run_agent(ctx, c, max_tool_calls=CFG["max_tool_calls"], max_model_turns=max_turns or CFG["max_model_turns"], repair_retries=0)
        status = r.status + (f" ({r.fallback_reason})" if r.fallback_reason else "")
        detail = str(r.fallback_detail)[:150] if r.fallback_reason == "api_failure" else ""
    except GroqError as e:
        status, detail = "ESCAPED", str(e)[:150]
    return {"as_of": as_of, "ratio": ratio, "sequential": sequential, "max_model_turns": max_turns or CFG["max_model_turns"], "calls_per_turn": [len(b) for b in batches], "status": status, "detail": detail, "requests": log,
            "refused": [x for x in c.request_log if x.get("refused_before_sending")], "total": sum(x["prompt_tokens"] + x["completion_tokens"] for x in log), "waited_s": round(c.pacer.waited, 1)}

out = [run(a, r, sequential=sq, max_turns=mt) for a in ("2025-10-13", "2025-11-10") for r in (2.6, 3.2, 3.8, 4.3, 4.9) for sq, mt in ((False, None), (True, None), (True, 20))]
print(json.dumps(out, indent=1))
