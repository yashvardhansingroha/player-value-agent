"""OFFLINE measurement of the model-visible text volume of one agent run (dev seed 1 only). No network, no model, no writes under the project.
Counts CHARACTERS of the exact serialised text; tokens are an ESTIMATE (no tokenizer for gpt-oss is available offline)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.run_agent import build_environment  # noqa: E402  (harness: dev seed 1, read-only)
from src.agent import agent as A  # noqa: E402
from src.agent import tools as T  # noqa: E402
from src.agent.llm import ToolCall  # noqa: E402
from src.ledger.proposal import jsonable  # noqa: E402

CASES = ["2025-10-13", "2025-11-10"]


def compact(env: dict, ndigits=None):
    """Candidate model-visible view: no whitespace; drop fields the model never needs to echo (output_sha256, per-envelope as_of and policy_hash, args duplicate);
    keep tool_call_id, tool, channel, populations and the full result."""
    keep = {k: v for k, v in env.items() if k not in ("output_sha256", "as_of", "policy_hash", "args")}
    if ndigits is not None:
        keep = round_floats(keep, ndigits)
    return json.dumps(keep, separators=(",", ":"), ensure_ascii=False)


def round_floats(o, nd):
    if isinstance(o, float):
        return float(f"{o:.{nd}g}")
    if isinstance(o, dict):
        return {k: round_floats(v, nd) for k, v in o.items()}
    if isinstance(o, list):
        return [round_floats(v, nd) for v in o]
    return o


out = {}
for as_of in CASES:
    ctx = build_environment(as_of)
    runner = T.ToolRunner(ctx, 200)
    rows = []

    def run(name, args):
        env = runner.execute(ToolCall(name, args))
        full = json.dumps(jsonable(env), indent=None)  # default separators, as a naive text serialisation
        rows.append({"tool": name, "channel": args.get("channel", "-"), "full": len(full), "compact": len(compact(env)), "compact_6sf": len(compact(env, 6)), "compact_4sf": len(compact(env, 4)), "error": "error" in env})
        return env

    run("get_review_overview", {})
    for ch in ctx.channels:
        run("get_policy_decision", {"channel": ch})
        run("get_quality_checks", {"channel": ch})
        run("get_forecast_and_populations", {"channel": ch})
        run("get_economics", {"channel": ch})
        d = ctx.decision(ch)
        if d["action"] == "increase_test":
            run("plan_experiment", {"channel": ch, "new_budget": round(d["proposed_budget"], 6)})
    run("check_budget_constraints", {"changes": [{"channel": c, "new_budget": ctx.budgets[c]} for c in ctx.channels]})
    out[as_of] = {"channels": ctx.channels, "actions": {c: ctx.decision(c)["action"] for c in ctx.channels}, "rows": rows}

decl = json.dumps(T.declarations(), separators=(",", ":"))
res = {"system_prompt_chars": len(A.SYSTEM_PROMPT), "tool_declarations_chars_compact_json": len(decl), "tool_declarations_chars_openai_wrapped": len(json.dumps([{"type": "function", "function": d} for d in T.declarations()], separators=(",", ":"))),
       "task_prompt_chars": len(A.task_prompt(ctx, 40)), "wrap_up_chars": len(A.WRAP_UP), "cases": out}
print(json.dumps(res, indent=1))
