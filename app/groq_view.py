"""Read-only view of the recorded Groq live runs (reports/phase5_runs/live_groq_*.json and the status files). ALL DATA IS SYNTHETIC. Pure JSON reading: no model call, no key, no provider code.

Used by the screen and the generated README so every Groq statement comes from the SAME verified files. `load()` re-hashes every recorded file against reports/groq_runs_manifest.json
(sha256 after CRLF normalisation); a mismatch is reported in `problems` and the callers must show it instead of the numbers.
"""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT  # manifest keys are paths relative to the repository root
MANIFEST = ROOT / "reports" / "groq_runs_manifest.json"
CACHE = ROOT / "reports" / "ui_review_cache.json"


def sha_lf(b: bytes) -> str:
    return hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()


def _note(run: dict) -> str:
    detail = str(run.get("fallback_detail") or "")
    m = re.search(r"attempted to call tool '([^']+)'", detail)
    if run["status"] == "agent_proposal":
        return f"validated by the unchanged validator; {run['repairs_used']} repair(s)"
    if m:
        return f"the provider rejected a model-generated call to the non-existent tool '{m.group(1)}' (HTTP 400 tool_use_failed)"
    if run.get("fallback_reason") == "malformed_output":
        return "the final answer was invalid JSON and was rejected again after one repair"
    return str(run.get("fallback_reason") or "")


def load():
    """None if the evidence is not present; otherwise a dict with `problems` (empty when every file matches its recorded hash)."""
    mp = MANIFEST
    if not mp.exists():
        return None
    man = json.loads(mp.read_text(encoding="utf-8"))
    problems = []
    for rel, e in man["files"].items():
        p = EVIDENCE / rel
        if not p.exists():
            problems.append(f"missing: {rel}")
        elif sha_lf(p.read_bytes()) != e["sha256_lf_normalised"]:
            problems.append(f"changed since recorded: {rel}")
    names = sorted(r for r in man["files"] if r.startswith("reports/phase5_runs/live_groq_"))
    runs = [json.loads((EVIDENCE / r).read_text(encoding="utf-8")) for r in names if (EVIDENCE / r).exists()]
    attempts = []
    for n, (name, run) in enumerate(zip(names, runs), 1):
        u = run.get("usage") or {}
        attempts.append({"attempt": n, "file": name, "flow": "staged (tools, then strict-schema finalization)" if run.get("finalization") == "staged_structured" else "single-stage (free-text JSON answer)",
                         "status": run["status"], "fallback_reason": run.get("fallback_reason"), "mode": run["mode"], "provider": run["provider"], "model": run["model"], "as_of": run["as_of"],
                         "prompt_tokens": u.get("prompt_tokens") or 0, "completion_tokens": u.get("output_tokens") or 0, "note": _note(run)})
    validated = [(a, r) for a, r in zip(attempts, runs) if a["status"] == "agent_proposal" and a["mode"] == "live"]
    comparison, run4 = [], None
    if validated and CACHE.exists():
        a, run4 = validated[0]
        cache = json.loads(CACHE.read_text(encoding="utf-8"))
        ref = {i["channel"]: i for i in cache["dates"][run4["as_of"]]["rules_only"]["proposal"]["items"]}
        for it in run4["proposal"]["items"]:
            r = ref[it["channel"]]
            comparison.append({"channel": it["channel"], "groq_action": it["action"], "groq_budget": it["new_budget"], "current_budget": it["current_budget"], "rules_action": r["action"], "rules_budget": r["new_budget"],
                               "claims_cited": len((it.get("evidence") or {}).get("claims", [])),
                               "match": it["action"] == r["action"] and abs(it["new_budget"] - r["new_budget"]) < 1e-6})
    stages = (run4 or {}).get("stages") or {}
    tools_used = sorted({c["tool"] for c in (run4 or {}).get("tool_calls", [])})
    return {"manifest": man, "problems": problems, "attempts": attempts, "n_attempts": len(attempts), "n_validated": len(validated), "n_fallback": sum(1 for a in attempts if a["status"] != "agent_proposal"),
            "comparison": comparison, "all_match": bool(comparison) and all(c["match"] for c in comparison), "stages": stages, "tools_used": tools_used,
            "validated_file": man.get("validated_run_file"),
            "validated_review": run4["as_of"] if run4 else None, "validated_model": run4["model"] if run4 else None,
            "summary_claims": len(json.loads([t for t in run4["model_turns"] if t.get("stage") == "finalization"][0]["text"])["summary_claims"]) if run4 else None,
            "tokens_validated": ((run4 or {}).get("usage") or {})}
