"""usage: python -I evals/groq_feasibility/revalidate_run.py live_groq_2025-11-10_3.json   (dev seed 1 data required)
OFFLINE re-validation of the model answers recorded in a live Groq run. No model call, no network. The deterministic tools are re-executed in the recorded order so the
evidence ids match; the recorded answer text is then given to the real validator."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.run_agent import build_environment  # noqa: E402  (dev seed 1)
from src.agent import tools as T  # noqa: E402
from src.agent import validator as V  # noqa: E402
from src.agent.llm import ToolCall  # noqa: E402

run = json.loads((ROOT / "reports" / "phase5_runs" / sys.argv[1]).read_text(encoding="utf-8"))
ctx = build_environment(run["as_of"])
turns = run["model_turns"]
runner = T.ToolRunner(ctx, 40)
n_calls_done = 0
for i, t in enumerate(turns, 1):
    for c in t["tool_calls"]:
        runner.execute(ToolCall(c["name"], c["args"]))
        n_calls_done += 1
    if t["text"]:
        print(f"=== model turn {i}: answer text, {len(t['text'])} chars, evidence available when it was written: tc_001..tc_{n_calls_done:03d}")
        out, errs = V.parse_output(t["text"])
        if out is None:
            print("   PARSE FAILED:", errs)
            bad = t["text"]
            pos = int(errs[0]["detail"].split("char ")[-1].rstrip(")")) if "char " in errs[0]["detail"] else None
            if pos is not None:
                print("   text around the failing character:", repr(bad[max(0, pos - 160):pos + 80]))
            print("   control characters in text:", [(k, hex(ord(ch))) for k, ch in enumerate(bad) if ord(ch) < 32][:10])
            continue
        rep = V.validate_agent_output(out, ctx, runner)
        print("   parsed OK; validator ok =", rep["ok"], "| errors:", len(rep["errors"]))
        for e in rep["errors"][:14]:
            print("     -", e["code"], e.get("channel") or "", "|", str(e["detail"])[:200])
