"""Phase 5 harness: run the deterministic pipeline FIRST, then the agent, in exactly one labelled mode.

    python scripts/run_agent.py --mode mock                      # scripted analyst, no network: plumbing demonstration (NOT agent evidence)
    python scripts/run_agent.py --mode replay --recording demo/recorded_run_2025-10-13.json
    python scripts/run_agent.py --mode live [--model ID]         # needs GEMINI_API_KEY; lists models, verifies one, runs the cases, records them

The key is read ONLY from the GEMINI_API_KEY environment variable, never printed, stored or committed. Without it, --mode live writes a PENDING status
and exits 0: live verification is then NOT done, and nothing is fabricated. Results go to reports/phase5_runs/<mode>_<as_of>.json (modes never mixed);
reports/phase5_report.md is regenerated from those files. ALL DATA IS SYNTHETIC; the experiment world is SIMULATED; the ledger is a SANDBOX.
Nothing in this script approves or applies anything.
"""
import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import agent as A  # noqa: E402
from src.agent import tools as T  # noqa: E402
from src.agent.groq_client import KEY_VAR as GROQ_KEY_VAR  # noqa: E402
from src.agent.groq_client import GroqClient  # noqa: E402
from src.agent.live import select_verified_model  # noqa: E402
from src.agent.llm import KEY_VAR, GeminiClient, PacedClient, RateLimiter, ReplayLLM, http_timeout_ms, recording_from_run  # noqa: E402
from src.agent.planning import make_real_context  # noqa: E402
from src.agent.scripted import scripted_llm  # noqa: E402
from src.experiments import design as D  # noqa: E402
from src.experiments.simulator import ExperimentSimulator, load_hidden, load_sim_config, value_reference  # noqa: E402  (hidden side, harness only)
from src.ledger.proposal import jsonable  # noqa: E402
from src.policy import constraints as cons  # noqa: E402
from src.policy import policy as pol  # noqa: E402

RUNS = ROOT / "reports" / "phase5_runs"
STATUS = ROOT / "reports" / "phase5_live_status.json"
GROQ_STATUS = ROOT / "reports" / "groq_live_status.json"  # Groq attempts are recorded here, never in the Gemini status file
DEMO = ROOT / "demo"
CASES = [("2025-10-13", "applovin: the policy proposes an increase test, but the experiment design is inadequate"),
         ("2025-11-10", "google_uac: the tracking outage window; the policy abstains (tracking gap)")]


def agent_cfg() -> dict:
    return yaml.safe_load((ROOT / "config" / "agent.yaml").read_text(encoding="utf-8"))


def build_environment(as_of: str, seed: int = 1):
    """The deterministic pipeline for one review (policy.decide_all) plus the real recalculation hooks. The experiment units' pre-period history comes from the
    SIMULATOR, which only this harness may import."""
    pcfg, fcfg, ecfg, sim_cfg = cons.load_policy(), cons.load_forecast_config(), D.load_config(), load_sim_config()
    truth = load_hidden(seed)
    dec = sqlite3.connect(ROOT / "data" / "decision" / f"seed_{seed}" / "decision.db")
    st = sqlite3.connect(ROOT / "data" / "forecasts" / f"seed_{seed}" / "forecasts.db")
    budgets = pol.initial_budgets_from_spend(dec, "2025-09-08")  # the same sandbox starting budgets as the Phase 3 and 4 reports
    ceiling = sum(budgets.values()) * (1 + pcfg["constraints"]["ceiling_reserve_fraction"])
    result = pol.decide_all(dec, st, as_of, budgets, ceiling, pcfg, fcfg)

    def provider(channel, units, win):
        base = ecfg["design"]["assignment_seed_base"]
        sim = ExperimentSimulator(truth, channel, {u["unit_id"]: u["baseline_daily_budget"] for u in units}, value_reference(dec, channel, as_of, sim_cfg["value_reference"]["matured_cohorts"]),
                                  sim_cfg, scale=1, unit_seed=D.derive_seed(base, "unit-truth", channel, as_of))
        return sim.pre_rows(win, D.derive_seed(base, "observations", channel, as_of))

    return make_real_context(dec, st, as_of, result, budgets, ceiling, 1, pcfg, fcfg, ecfg, provider)


def save_run(run, tag: str) -> Path:
    """Mock and replay runs are regenerated in place. A LIVE run is never overwritten: every attempt keeps its own file (live_<date>.json, then _2, _3, ...)."""
    RUNS.mkdir(parents=True, exist_ok=True)
    p = RUNS / f"{tag}_{run.as_of}.json"
    if tag == "live":
        n = 1
        while p.exists():
            n += 1
            p = RUNS / f"{tag}_{run.as_of}_{n}.json"
    p.write_text(json.dumps(run.to_dict(), indent=1), encoding="utf-8")
    return p


def run_case(ctx, llm, cfg):
    return A.run_agent(ctx, llm, max_tool_calls=cfg["max_tool_calls"], max_model_turns=cfg.get("max_model_turns", 30), repair_retries=cfg["repair_retries"])


def summary_line(run) -> str:
    return f"{run.as_of} [{run.mode}] {run.status}" + (f" ({run.fallback_reason})" if run.fallback_reason else "") + f"; tool calls {len(run.tool_calls)}; repairs {run.repairs_used}"


def mode_mock(cfg):
    for as_of, _ in CASES:
        ctx = build_environment(as_of)
        run = run_case(ctx, scripted_llm(ctx), cfg)
        print(save_run(run, "mock").name, "->", summary_line(run), flush=True)


def mode_replay(cfg, path):
    llm = ReplayLLM.from_file(path)
    ctx = build_environment(llm.recording["as_of"])
    run = run_case(ctx, llm, cfg)
    print(save_run(run, "replay").name, "->", summary_line(run), flush=True)


def select_cases(only):
    """All cases, or only the listed review dates (so a small daily quota can be spent over two runs)."""
    if not only:
        return list(CASES)
    unknown = sorted(set(only) - {d for d, _ in CASES})
    if unknown:
        raise SystemExit(f"unknown case date(s) {unknown}; the cases are {[d for d, _ in CASES]}")
    return [c for c in CASES if c[0] in set(only)]


def mode_live(cfg, requested, only=None):
    cases = select_cases(only)
    key = os.environ.get(KEY_VAR)
    if not key:
        STATUS.write_text(json.dumps({"label": "LIVE VERIFICATION PENDING", "key_present": False, "verified_model": None, "connectivity_check": "NOT RUN",
                                      "live_agent_runs": [], "note": f"{KEY_VAR} is not set in the environment of this process. Nothing was called and nothing is recorded; "
                                                                      "set the variable and re-run `python scripts/run_agent.py --mode live`."}, indent=1), encoding="utf-8")
        print(f"PENDING: {KEY_VAR} is not set. Live Gemini connectivity and the live agent runs are NOT verified.")
        return 0
    from google import genai
    from google.genai import types
    # ONE limiter paces the model-selection calls and the agent's calls, so the per-minute quota is respected across both
    limiter = RateLimiter(cfg.get("min_seconds_between_requests", 0), cfg.get("max_rate_limit_retries", 2), cfg.get("max_rate_limit_wait_seconds", 90))
    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=http_timeout_ms(cfg["timeout_seconds"])))  # configured seconds -> SDK milliseconds
    sel = select_verified_model(PacedClient(client, limiter), requested)
    # earlier attempts are kept: the previous status (if it recorded a real attempt) moves into `history`; nothing is overwritten
    old = json.loads(STATUS.read_text(encoding="utf-8-sig")) if STATUS.exists() else {}
    history = list(old.get("history", []))
    if old.get("key_present"):
        history.append({"attempt": old.get("attempt", len(history) + 1), "verified_model": old.get("verified_model"), "connectivity_check": old.get("connectivity_check"),
                        "live_agent_runs": old.get("live_agent_runs", [])})
    status = {"label": "LIVE VERIFICATION", "attempt": len(history) + 1, "key_present": True, "listed_models": sel["listed"], "models_tried": sel["tried"],
              "selection_stopped": sel.get("stopped"), "verified_model": sel["verified_model"], "connectivity_check": "PASSED (plain, tool call and round trip)" if sel["verified_model"] else "FAILED",
              "live_agent_runs": [], "history": history}

    def write_status():  # written after every step, so an interrupted run still leaves its record
        STATUS.write_text(json.dumps(jsonable(status), indent=1).replace(key, "<key>"), encoding="utf-8")

    write_status()
    if not sel["verified_model"]:
        print("FAILED: no listed model passed the live round-trip check; see reports/phase5_live_status.json")
        if sel.get("stopped"):
            print(sel["stopped"])
        return 1
    print("verified model:", sel["verified_model"], flush=True)
    for as_of, why in cases:
        ctx = build_environment(as_of)
        run = run_case(ctx, GeminiClient(sel["verified_model"], temperature=cfg.get("temperature", 0.0), timeout_seconds=cfg["timeout_seconds"], limiter=limiter), cfg)
        p = save_run(run, "live")
        status["live_agent_runs"].append({"as_of": as_of, "case": why, "status": run.status, "fallback_reason": run.fallback_reason, "file": str(p.relative_to(ROOT)), "repairs_used": run.repairs_used})
        if run.mode == "live" and run.status == "agent_proposal":  # only an actual, validated live run is saved as a replay recording
            DEMO.mkdir(exist_ok=True)
            (DEMO / f"recorded_run_{p.stem[len('live_'):]}.json").write_text(json.dumps(recording_from_run(run.to_dict()), indent=1), encoding="utf-8")
        write_status()
        print(summary_line(run), flush=True)
    return 0


def mode_live_groq(only=None):
    """Groq live runs. No model listing: the model id comes from config/agent_groq.yaml, and the first request is the first real check of it. A client
    built here reads GROQ_API_KEY itself, so its mode is 'live'; each case's request log (estimated vs actual tokens, rate-limit headers, waits) is saved with the run."""
    gcfg = yaml.safe_load((ROOT / "config" / "agent_groq.yaml").read_text(encoding="utf-8"))
    cases = select_cases(only)
    if not os.environ.get(GROQ_KEY_VAR):
        GROQ_STATUS.write_text(json.dumps({"label": "GROQ LIVE VERIFICATION PENDING", "provider": "groq", "model": gcfg["model"], "key_present": False, "live_agent_runs": [],
                                           "note": f"{GROQ_KEY_VAR} is not set in the environment of this process. Nothing was called and nothing is recorded."}, indent=1), encoding="utf-8")
        print(f"PENDING: {GROQ_KEY_VAR} is not set. Live Groq runs are NOT done.")
        return 0
    client = GroqClient(gcfg["model"], gcfg)
    old = json.loads(GROQ_STATUS.read_text(encoding="utf-8-sig")) if GROQ_STATUS.exists() else {}
    history = list(old.get("history", []))  # earlier attempts are kept, as for Gemini: nothing is overwritten
    if old.get("key_present"):
        history.append({"attempt": old.get("attempt", len(history) + 1), "live_agent_runs": old.get("live_agent_runs", [])})
    status = {"label": "GROQ LIVE ATTEMPT", "attempt": len(history) + 1, "provider": "groq", "model": gcfg["model"], "key_present": True, "live_agent_runs": [], "history": history}
    for as_of, why in cases:
        ctx = build_environment(as_of)
        start = len(client.request_log)
        run = A.run_agent(ctx, client, max_tool_calls=gcfg["max_tool_calls"], max_model_turns=gcfg["max_model_turns"], repair_retries=gcfg["repair_retries"])
        d = run.to_dict()
        d["request_log"] = client.request_log[start:]
        RUNS.mkdir(parents=True, exist_ok=True)
        p, n = RUNS / f"live_groq_{as_of}.json", 1
        while p.exists():
            n += 1
            p = RUNS / f"live_groq_{as_of}_{n}.json"
        p.write_text(json.dumps(d, indent=1), encoding="utf-8")
        status["live_agent_runs"].append({"as_of": as_of, "case": why, "status": run.status, "fallback_reason": run.fallback_reason, "file": str(p.relative_to(ROOT)), "repairs_used": run.repairs_used,
                                          "tokens": run.usage})
        if run.mode == "live" and run.status == "agent_proposal" and run.finalization is None:  # only an actual, validated live run becomes a replay recording (staged runs have two stages; replay does not support them yet)
            DEMO.mkdir(exist_ok=True)
            (DEMO / f"recorded_run_groq_{p.stem[len('live_groq_'):]}.json").write_text(json.dumps(recording_from_run(d), indent=1), encoding="utf-8")
        GROQ_STATUS.write_text(json.dumps(jsonable(status), indent=1), encoding="utf-8")
        print(summary_line(run), f"[provider {run.provider}, tokens {run.usage}]", flush=True)
        if run.stages:  # staged run: report the stages separately
            i, f = run.stages["investigation"], run.stages["finalization"]
            print(f"  investigation: ended_by={i['ended_by']} required_evidence_complete={i['required_evidence_complete']} model_requests={i['model_requests']} nudges={i['evidence_nudges']} tokens={i['usage']}")
            print(f"  finalization:  requests={f['requests']} provider_accepted_schema_and_returned_text={f['provider_accepted_schema_and_returned_text']} json={f['text_parsed_as_json_object']} validator_ok={f['validator_ok']} repairs={f['repairs_used']} tokens={f['usage']}")
    return 0


# ---------------------------------------------------------------------------------------------- report


def md_table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(r.get(c, "")).replace("|", "/").replace("\n", " ") for c in cols) + " |" for r in rows]
    return "\n".join(out)


def write_report():
    status = json.loads(STATUS.read_text(encoding="utf-8-sig")) if STATUS.exists() else {"label": "LIVE VERIFICATION PENDING", "key_present": False, "note": "no live attempt recorded"}
    runs = [dict(json.loads(p.read_text(encoding="utf-8")), _file=p.name) for p in sorted(RUNS.glob("*.json"))] if RUNS.exists() else []
    L = ["# Phase 5 report (generated by scripts/run_agent.py)", "",
         "**ALL DATA IS SYNTHETIC. Experiments are SIMULATED. The ledger is a SANDBOX. The agent proposes; it cannot approve or apply anything.**", "",
         "## 1. Live verification status (kept apart from mock and replay)", "",
         f"- **{status['label']}**: key present = {status.get('key_present')}; connectivity check = {status.get('connectivity_check', 'NOT RUN')}; verified model = {status.get('verified_model')}.",
         f"- {status.get('note', '')}", f"- live agent runs in the latest attempt: {len(status.get('live_agent_runs', []))}"]
    for h in status.get("history", []):  # earlier live attempts are never hidden
        L.append(f"- earlier attempt {h.get('attempt')}: " + "; ".join(f"{r['as_of']} {r['status']}" + (f" ({r['fallback_reason']})" if r.get("fallback_reason") else "") for r in h.get("live_agent_runs", [])))
    L += ["",
         "Mock and replay results below are plumbing demonstrations. **Only a passing live run with a real key is evidence that Gemini can do this task, and even then one run is not an evaluation (Phase 6).**", "",
         "## 2. Tool contracts (read-only; no SQL, Python, shell, file or ledger-execution tool)", "",
         md_table([{"tool": t.name, "arguments": ", ".join(f"{k}:{v['type']}" for k, v in t.parameters.get("properties", {}).items()) or "none",
                    "returns": t.description, "population labels": "; ".join([t.default_population] + sorted(set(t.populations.values())))} for t in T.TOOLS],
                  ["tool", "arguments", "returns", "population labels"]), "",
         "Every successful call is wrapped in an envelope {tool_call_id, tool, args, channel, as_of, policy_hash, populations, result, output_sha256}. A claim must name (tool_call_id, field path in `result`, value, population, channel) and is checked against the stored envelope.", ""]
    for r in runs:
        L += [f"## Run: {r['mode'].upper()} {r['as_of']} ({r['status']}) [{r['_file']}]", "", f"- model: `{r['model']}`; repairs used: {r['repairs_used']}; fallback: {r['fallback_reason']}; tool calls: {len(r['tool_calls'])}; policy hash `{r['policy_hash'][:16]}...`"]
        L += [f"- {l}" for l in r["labels"]] + [""]
        if r["proposal"]:
            L += [md_table([{"channel": i["channel"], "action": i["action"], "budget": f"{i['current_budget']:g} -> {i['new_budget']:g}", "qualification": i["qualification"],
                             "plan": (i["experiment_plan_hash"] or "")[:12], "reason": i["reason"][:230]} for i in r["proposal"]["items"]], ["channel", "action", "budget", "qualification", "plan", "reason"]), ""]
        L += ["Evidence trail:", "", md_table([{"id": c["tool_call_id"], "tool": c["tool"], "args": json.dumps(c["args"]), "status": c["status"]} for c in r["tool_calls"]], ["id", "tool", "args", "status"]), ""]
        v = r.get("validation") or r.get("fallback_validation")
        L += [f"Validator: ok = {v['ok'] if v else None}; errors = {[e['code'] for e in (v or {}).get('errors', [])]}", ""]
    (ROOT / "reports" / "phase5_report.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mock", "replay", "live"], required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--provider", choices=["gemini", "groq"], default="gemini", help="live mode only; groq reads GROQ_API_KEY and config/agent_groq.yaml")
    ap.add_argument("--recording", default=None)
    ap.add_argument("--only", action="append", default=None, help="live mode: run only this review date (repeatable), e.g. --only 2025-11-10; the connectivity check always runs first")
    args = ap.parse_args()
    cfg = agent_cfg()
    code = 0
    if args.mode == "mock":
        mode_mock(cfg)
    elif args.mode == "replay":
        if not args.recording:
            sys.exit("--recording is required for replay")
        mode_replay(cfg, args.recording)
    elif args.provider == "groq":
        code = mode_live_groq(args.only)
    else:
        code = mode_live(cfg, args.model, args.only)
    write_report()
    return code


if __name__ == "__main__":
    sys.exit(main())
