"""Logic behind the Monday-review screen (app/main.py): pure functions over the pre-generated cache and the sandbox ledger. ALL DATA IS SYNTHETIC; experiments are SIMULATED;
the ledger is a SANDBOX. This module never calls a model, never reads the Gemini key and never touches hidden truth: the screen shows results that scripts already produced
(reports/ui_review_cache.json from scripts/build_ui_data.py, plus the committed Phase 4/5/6 result files) and lets a PERSON approve, apply, reject and roll back in a sandbox ledger.
"""
import json
import tempfile
from pathlib import Path

import pandas as pd

from app import groq_view
from src.ledger.ledger import Ledger
from src.ledger.proposal import Proposal
from src.policy import constraints as cons

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "reports" / "ui_review_cache.json"
SOURCES = {"rules_only": "Rules-only baseline (deterministic, no agent)", "agent_mock": "MOCK agent (scripted; plumbing test, NOT a live model)"}


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig")) if Path(path).exists() else None


def load_cache():
    return read_json(CACHE)


def cache_status(cache) -> tuple:
    """(ok, message). The cache is only shown if it was built under the CURRENT policy/forecast configuration."""
    if cache is None:
        return False, "reports/ui_review_cache.json is missing. Generate it with `python scripts/build_ui_data.py` (needs the generated dev data)."
    current = cons.policy_hash(cons.load_policy(), cons.load_forecast_config())
    if cache.get("policy_hash") != current:
        return False, "The review cache was built under a DIFFERENT policy/forecast configuration than the current one, so it is stale. Regenerate it with `python scripts/build_ui_data.py`."
    return True, "ok"


def review_dates(cache) -> list:
    return sorted(cache["dates"])


def review_table(cache, day: str) -> pd.DataFrame:
    rows = []
    for ch, v in cache["dates"][day]["channels"].items():
        rows.append({"channel": ch, "data ready": "yes" if v["completeness_passed"] else ("no" if v["completeness_passed"] is False else "n/a"),
                     "failed gates": sum(1 for s in v["stages"] if not s["passed"]), "policy action": v["policy_action"], "current budget": v["current_budget"],
                     "policy budget": v["policy_budget"], "why": (v["reasons"][0] if v["reasons"] else "")[:140]})
    return pd.DataFrame(rows)


def forecast_table(cache, day: str) -> pd.DataFrame:
    rows = []
    for ch, v in cache["dates"][day]["channels"].items():
        f, a, t = v["forecast"], v["average_d30_roas"], v["payback_target"]
        iv = (f or {}).get("interval") or {}
        rows.append({"channel": ch, "forecast D30 net revenue (included cohorts only)": None if not f else f["forecast_d30_net_revenue"],
                     "interval low": iv.get("lo"), "interval high": iv.get("hi"), "interval": iv.get("interval_status", "n/a"),
                     "avg D30 ROAS": (a or {}).get("point"), "ROAS low": (a or {}).get("lower"), "ROAS high": (a or {}).get("upper"),
                     "payback target": (t or {}).get("target_d30_roas"),
                     "marginal ROAS (model-estimated step)": None if not v["marginal_step"] else v["marginal_step"]["marginal_roas"],
                     "marginal lower bound": None if not v["marginal_step"] else v["marginal_step"]["lower_bound"],
                     "included cohorts": None if not f else f["included_cohorts"], "withheld cohorts": None if not f else f["excluded_withheld_cohorts"]})
    return pd.DataFrame(rows)


def stages_table(cache, day: str, channel: str) -> pd.DataFrame:
    return pd.DataFrame([{"gate": s["stage"], "passed": "pass" if s["passed"] else "FAIL", "if it fails it blocks": s["blocks"], "detail": s["detail"]}
                         for s in cache["dates"][day]["channels"][channel]["stages"]])


def get_source(cache, day: str, source: str) -> dict:
    """The proposal source for a review date. Returns {source, labels, proposal (dict), validation, tool_calls, evidence_trail, status, note}."""
    d = cache["dates"][day]
    if source == "rules_only":
        r = d["rules_only"]
        return {"source": source, "title": SOURCES[source], "labels": r["labels"], "proposal": r["proposal"], "validation": r["validation"], "tool_calls": [], "evidence_trail": [],
                "status": "rules_only_baseline", "note": "Deterministic policy output after the experiment-design gate; it passes the same validator layer as an agent proposal."}
    r = d["agent_mock"]
    return {"source": source, "title": SOURCES[source], "labels": r["labels"], "proposal": r["proposal"], "validation": r.get("validation") or r.get("fallback_validation"), "tool_calls": r["tool_calls"],
            "evidence_trail": r["evidence_trail"], "status": r["status"], "note": "A SCRIPTED analyst that follows the policy by construction. It shows the tools, citations and validator working on real review data; it is not evidence about any model."}


def proposal_items_table(prop: dict) -> pd.DataFrame:
    return pd.DataFrame([{"channel": i["channel"], "action": i["action"], "budget": f"{i['current_budget']:g} -> {i['new_budget']:g}", "qualification": i["qualification"],
                          "plan": (i["experiment_plan_hash"] or "")[:12], "reason": i["reason"]} for i in prop["items"]])


def claims_table(item: dict) -> pd.DataFrame:
    return pd.DataFrame(item.get("evidence", {}).get("claims", []))


def trail_table(src: dict) -> pd.DataFrame:
    return pd.DataFrame([{"id": c["tool_call_id"], "tool": c["tool"], "args": json.dumps(c["args"]), "status": c["status"]} for c in src["tool_calls"]])


# ---------------------------------------------------------------------------------------------- sandbox ledger (a PERSON acts here, never the agent)


def new_sandbox(cache) -> dict:
    """A fresh sandbox ledger file. Only its PATH is kept in the session: Streamlit runs every rerun in its own thread and a SQLite connection cannot cross threads, so each run
    opens its own connection with open_ledger()."""
    first = cache["dates"][review_dates(cache)[0]]
    folder = tempfile.mkdtemp(prefix="sandbox_ledger_")
    path = str(Path(folder) / "ledger.db")
    led = Ledger(path, cons.load_policy(), cons.load_forecast_config())
    led.initialize(first["budgets"], first["ceiling"], now=review_dates(cache)[0])
    led.close()
    return {"path": path, "folder": folder}


def open_ledger(sandbox: dict) -> Ledger:
    """A connection to the sandbox ledger for THIS run (same configuration as the one it was initialised with)."""
    return Ledger(sandbox["path"], cons.load_policy(), cons.load_forecast_config())


def proposal_status(led: Ledger, proposal_id: str):
    r = led.conn.execute("SELECT status FROM proposals WHERE proposal_id = ?", (proposal_id,)).fetchone()
    return None if r is None else r[0]


def submit(led: Ledger, prop: dict, now: str) -> dict:
    """Record the proposal in the ledger (it can only be built on the CURRENT ledger version; anything older is refused as stale)."""
    p = Proposal.model_validate(prop)
    if proposal_status(led, p.proposal_id) is not None:
        return {"ok": False, "code": "already_submitted", "message": f"proposal {p.proposal_id} is already in the ledger (status {proposal_status(led, p.proposal_id)})."}
    if p.ledger_version != led.head():
        return {"ok": False, "code": "stale_ledger_version", "message": f"This proposal was built on ledger version {p.ledger_version} but the ledger is at version {led.head()}. "
                "A real review would be re-run on the current ledger; here it is refused, which is the ledger's own safeguard."}
    if p.policy_hash != led.policy_hash:
        return {"ok": False, "code": "stale_policy_config", "message": "the proposal was built under a different policy configuration than this ledger holds."}
    led.create_proposal(p, now)
    return {"ok": True, "code": "submitted", "message": f"proposal {p.proposal_id} submitted (status proposed)."}


def decide(led: Ledger, proposal_id: str, action: str, approver: str, now: str) -> dict:
    if action == "approve":
        r = led.approve(proposal_id, approver, now)
    elif action == "reject":
        r = led.reject(proposal_id, approver, "rejected in the review screen", now)
    elif action == "apply":
        r = led.apply(proposal_id, now)
    else:
        raise ValueError(action)
    ok = r.get("ok", False)
    return {"ok": ok, "code": r.get("code", action), "message": (f"{action}: done." if ok else f"{action} REFUSED: {r.get('code')} {json.dumps(r.get('detail', {}), default=str)[:300]}"), "raw": r}


def active_changes(led: Ledger) -> list:
    return [{"change_id": r[0], "kind": r[1], "proposal_id": r[2], "applied_at": r[3]} for r in
            led.conn.execute("SELECT change_id, kind, proposal_id, applied_at FROM applied_changes WHERE change_type = 'APPLY' AND status = 'active' ORDER BY applied_at")]


def rollback(led: Ledger, change_id: str, who: str, now: str) -> dict:
    r = led.rollback(change_id, who, "rolled back in the review screen", now)
    ok = r.get("ok", False)
    return {"ok": ok, "code": r.get("status"), "message": ("rolled back." if ok else f"rollback {r.get('status', 'refused').upper()}: {r.get('code')} {json.dumps(r.get('detail', {}), default=str)[:300]}")}


def ledger_versions(led: Ledger) -> pd.DataFrame:
    return pd.DataFrame(led.conn.execute("SELECT version, kind, change_id, proposal_id, ceiling, note FROM ledger_versions ORDER BY version").fetchall(),
                        columns=["version", "kind", "change_id", "proposal_id", "ceiling", "note"])


def ledger_budgets(led: Ledger) -> pd.DataFrame:
    s = led.state()
    return pd.DataFrame([{"channel": c, "budget": b} for c, b in s["budgets"].items()] + [{"channel": "TOTAL", "budget": s["total"]}, {"channel": "ceiling", "budget": s["ceiling"]},
                                                                                         {"channel": "reserve (unallocated)", "budget": s["reserve"]}])


def audit_table(led: Ledger) -> pd.DataFrame:
    return pd.DataFrame([{"id": a["id"], "at": a["at"], "event": a["event"], "proposal": a["proposal_id"], "change": a["change_id"], "details": json.dumps(a["details"], default=str)[:200]} for a in led.audit()])


# ---------------------------------------------------------------------------------------------- committed result files shown on the screen


def experiment_view() -> dict:
    p4 = read_json(ROOT / "reports" / "phase4_results.json")
    if p4 is None:
        return {}
    life = p4.get("lifecycle") or {}
    return {"channel": p4["channel"], "as_of": p4["as_of"], "dev_world": pd.DataFrame(p4["scale_search"]), "demo_world": pd.DataFrame(p4.get("demo_search") or []),
            "transitions": pd.DataFrame(life.get("transitions", [])), "readout": life.get("readout"), "plan_hash": life.get("plan_hash"), "chosen_scale": p4.get("chosen_scale"),
            "score": p4.get("score"), "replicate": p4.get("replicate_study_chosen_design")}


def evaluation_view() -> dict:
    ev = read_json(ROOT / "evals" / "results" / "dev_eval.json")
    if ev is None:
        return {}
    f = next(r for r in ev["forecasting"]["table"] if r["level"] == "cohort" and r["group"] == "overall")
    return {"policy_overall": ev["policy"]["overall"], "per_scenario": pd.DataFrame(ev["policy"]["per_scenario"]),
            "violations": pd.DataFrame(ev["policy"]["forbidden_or_unlisted"])[["scenario", "channel", "date", "action", "ideal", "forbidden", "reason"]],
            "constraints": pd.DataFrame(ev["constraints"])[["id", "case", "layer", "expected", "observed", "refused_as_expected"]], "forecast_overall": f,
            "abstention": ev["policy"]["abstention"], "live": ev["live"], "held_out": ev["held_out"]}


def live_status() -> dict:
    s = read_json(ROOT / "reports" / "phase5_live_status.json") or {}
    attempts = list(s.get("history", [])) + ([{"attempt": s.get("attempt"), "verified_model": s.get("verified_model"), "connectivity_check": s.get("connectivity_check"),
                                               "live_agent_runs": s.get("live_agent_runs", []), "stopped": s.get("selection_stopped")}] if s else [])
    valid = [r for a in attempts for r in a.get("live_agent_runs", []) if r.get("status") == "agent_proposal"]
    return {"status": "PENDING" if not valid else "HAS VALID LIVE RUN(S)", "attempts": attempts, "valid_live_agent_runs": len(valid)}


def heldout_view() -> dict:
    """The held-out DETERMINISTIC-BASELINE state, read from files: 'untouched' (no manifest, no run lock), 'frozen_not_run' or 'run'. Rows are the stored per-seed results."""
    base = ROOT / "evals" / "results" / "heldout"
    man, lock = read_json(ROOT / "evals" / "FREEZE_MANIFEST.json"), read_json(base / "RUN_LOCK.json")
    state = "run" if lock else ("frozen_not_run" if man else "untouched")
    rows, failed = [], {}
    for d in sorted(base.glob("seed_*")):
        seed = int(d.name.split("_")[1])
        res, prov = read_json(d / "baseline_eval.json"), read_json(d / "provenance.json") or {"steps": {}}
        bad = {n: v.get("error") for n, v in prov["steps"].items() if v["status"] != "completed"}
        if bad:
            failed[seed] = bad
        if res:
            c = next(x for x in res["forecasting"]["table"] if x["level"] == "cohort" and x["group"] == "overall")
            o, a, v = res["policy_primary"]["overall"], res["policy_primary"]["abstention"], res["scenario_validity"]
            rows.append({"seed": seed, "cohort WAPE": c["wape"], "coverage (nominal 0.80)": c["coverage"], "mean width / forecast": c["interval_width_over_forecast_mean"], "scored reviews": o["n_scored"],
                         "ideal": o["ideal"], "acceptable": o["acceptable"], "forbidden": o["forbidden"], "unlisted": o["unlisted"], "abstention recall": a["recall"],
                         "intents holding": f"{v['n_pass']}/{v['n_intents']}", "scenarios not materialized": ", ".join(v["scenarios_not_materialized"]) or "none"})
    return {"state": state, "freeze_id": (man or {}).get("freeze_id"), "rows": pd.DataFrame(rows), "failed": failed}


def groq_summary():
    """The recorded Groq live runs (data copies, hash-checked), or None if the evidence is absent or does not match its manifest."""
    g = groq_view.load()
    return g if g and not g["problems"] else None


def status_text(live: dict, held: dict, groq=None) -> str:
    """One statement that keeps four kinds of result apart: (a) development, (b) held-out deterministic baseline, (c) mock agent plumbing, (d) live agent verification (Gemini blocked; recorded Groq runs)."""
    if held["state"] == "untouched":
        b = "(b) Held-out deterministic-baseline results: none yet; the held-out seeds are untouched and nothing has been frozen."
    elif held["state"] == "frozen_not_run":
        b = f"(b) Held-out deterministic-baseline results: not run yet; the baseline is frozen as {held['freeze_id']} and the held-out seeds are untouched."
    else:
        b = (f"(b) Held-out deterministic-baseline results ({held['freeze_id']}, run once): they validate ONLY the frozen rules-only baseline on additional synthetic seeds, say nothing about any agent, "
             "and those seeds can no longer be called untouched holdouts for one.")
    d = (f"(d) Live agent verification: Gemini {live['status']} - {live['valid_live_agent_runs']} valid live-model run(s) recorded; its attempts are blocked by a recorded billing state (HTTP 402) after earlier rate-limit and quota stops.")
    if groq:
        d += (f" Groq ({groq['validated_model']}, see the README): {groq['n_validated']} validated live proposal on one synthetic review out of {groq['n_attempts']} full-case attempts, equal to the rules-only proposal; "
              f"the other {groq['n_fallback']} fell back. No reliability or decision-quality improvement is demonstrated. The proposals on this screen are rules-only or MOCK, not that live run.")
    return f"(a) Development results (seed 1) are in the evaluation tab. {b} (c) Agent results on this screen are MOCK plumbing tests, never evidence about a real model. {d}"

