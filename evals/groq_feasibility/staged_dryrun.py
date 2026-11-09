"""OFFLINE dry run of the two-stage flow on the real dev contexts (seed 1). NO network, NO model: the 'model' is a TARGETED scripted investigation (overview, the five policy decisions, the plan for a retrieved increase) and a minimal answer that follows the policy, delivered through a FAKED transport
whose usage figures assume 4.3 characters per token (the figure measured live on tool-result JSON). The real tools, evidence index, strict schema, validator and the real 8,000 tokens/min pacing run.
It shows request sizes and that the staged plumbing validates on real evidence; it says NOTHING about any model.   usage: python -I evals/groq_feasibility/staged_dryrun.py > staged_dryrun.json"""
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.run_agent import build_environment  # noqa: E402  (dev seed 1)
from src.agent import agent as A  # noqa: E402
from src.agent.groq_client import GroqClient, GroqError, HttpResult  # noqa: E402
from src.agent.llm import ToolCall  # noqa: E402
from src.agent.scripted import _claim  # noqa: E402

CFG = yaml.safe_load((ROOT / "config" / "agent_groq.yaml").read_text(encoding="utf-8"))
RATIO = 4.3


class Clock:
    t = 0.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


def targeted(ctx):
    """The calls a careful model makes (overview, every policy decision, the plan for each retrieved increase) and an answer that follows the policy after the experiment gate, with cited claims.
    Reasons contain no digits, so the numeric-consistency check has nothing to object to."""
    calls = [ToolCall("get_review_overview", {}, None)] + [ToolCall("get_policy_decision", {"channel": c}, None) for c in ctx.channels]
    for ch in ctx.channels:
        if ctx.decision(ch)["action"] == "increase_test":
            calls.append(ToolCall("plan_experiment", {"channel": ch, "new_budget": round(ctx.decision(ch)["proposed_budget"], 6)}, None))
    ids = {(c.name, c.args.get("channel")): f"tc_{n:03d}" for n, c in enumerate(calls, 1)}
    items = []
    for ch in ctx.channels:
        d, ref = ctx.decision(ch), ctx.reference(ch)
        claims = [_claim(ids[("get_policy_decision", ch)], "action", d["action"], "policy_decision", ch)]
        item = {"channel": ch, "action": ref["action"], "new_budget": ref["budget"], "qualification": "none", "experiment_plan_hash": None, "claims": claims, "reason": "The deterministic policy action is followed."}
        if d["action"] == "increase_test":
            q = ctx.qualification(ch, d["proposed_budget"])
            claims.append(_claim(ids[("plan_experiment", ch)], "qualifies", bool(q["qualifies"]), "experiment_plan_for_exact_step", ch))
            if q["qualifies"]:
                item.update(qualification="experiment_backed", experiment_plan_hash=q["plan_hash"])
            else:
                item["reason"] = "The experiment design does not qualify, so the evidence is insufficient and the budget is unchanged."
        items.append(item)
    return calls, json.dumps({"summary": "Every action follows the deterministic policy after the experiment gate.", "summary_claims": [], "items": items})


def run(as_of, sequential):
    ctx = build_environment(as_of)
    calls, answer_text = targeted(ctx)
    batches = [[c] for c in calls] if sequential else [calls]
    ck, log = Clock(), []

    def transport(body, timeout):
        n = len(log)
        chars = len(json.dumps({"messages": body["messages"], "tools": body.get("tools", []), "response_format": body.get("response_format")}, separators=(",", ":"), ensure_ascii=False))
        prompt = int(chars / RATIO)
        structured = "response_format" in body
        if structured:
            msg, out, kind = {"content": answer_text}, 900, "finalization"
        elif n < len(batches):
            msg = {"content": None, "tool_calls": [{"id": f"call_{n}_{i}", "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.args)}} for i, c in enumerate(batches[n])]}
            out, kind = 60 + 30 * len(batches[n]), "investigation"
        else:
            msg, out, kind = {"content": "DONE"}, 5, "investigation"
        log.append({"stage": kind, "t": round(ck.t, 1), "prompt_tokens": prompt, "completion_tokens": out, "max_completion_tokens": body["max_completion_tokens"]})
        return HttpResult(200, {}, json.dumps({"choices": [{"message": msg, "finish_reason": "stop"}], "usage": {"prompt_tokens": prompt, "completion_tokens": out, "total_tokens": prompt + out}}))

    c = GroqClient(CFG["model"], {**CFG, "assumed_chars_per_token": 4.0}, transport=transport, clock=ck.now, sleep=ck.sleep)
    try:
        r = A.run_agent(ctx, c, max_tool_calls=CFG["max_tool_calls"], max_model_turns=CFG["max_model_turns"], repair_retries=CFG["repair_retries"])
        status = r.status + (f" ({r.fallback_reason}: {str(r.fallback_detail)[:200]})" if r.fallback_reason else "")
        stages = r.stages
    except GroqError as e:
        status, stages = f"ESCAPED {e}", None
    return {"as_of": as_of, "investigation_calls": len(calls), "sequential_single_call_turns": sequential, "status": status, "stages": stages, "requests": log,
            "refused": [x for x in c.request_log if x.get("refused_before_sending")], "total_tokens": sum(x["prompt_tokens"] + x["completion_tokens"] for x in log), "seconds_waited": round(c.pacer.waited, 1)}


print(json.dumps([run(a, sq) for a in ("2025-10-13", "2025-11-10") for sq in (False, True)], indent=1))
