"""Run the FROZEN deterministic baseline on the registered held-out seeds, ONCE (evals/heldout_protocol.yaml). ALL DATA IS SYNTHETIC. No agent, no model call.

    python scripts/run_heldout_baseline.py --rehearse    # development seed 1 into a temporary folder; checks the harness reproduces the dev numbers. Allowed before the freeze.
    python scripts/run_heldout_baseline.py --execute     # seeds 9101, 9102, 9103, once. Refuses unless the freeze verifies and no earlier run exists.
    python scripts/run_heldout_baseline.py --execute --resume   # continue steps that did not complete, with the identical frozen code; recorded as a resume

This is the ONLY script that passes allow_heldout=True to the generator. It never calls scripts/run_backtest.py (that script overwrites the development reports).
Per seed it builds data, builds the forecast store (run_backtest, issued once), checks every artifact is tied to the seed, and only then computes results. A step that fails
is recorded with its error and the other seeds still run; nothing is retried and no failed output is deleted.
"""
import argparse
import json
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals import baseline_eval as B  # noqa: E402
from evals import dev_eval as E  # noqa: E402
from evals import freeze as F  # noqa: E402
from scripts.generate_data import build  # noqa: E402
from src.forecast.pipeline import load_config, run_backtest  # noqa: E402

RES = ROOT / "evals" / "results" / "heldout"
LOCK, LOG = RES / "RUN_LOCK.json", RES / "RUN_LOG.json"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_event(**kw):
    RES.mkdir(parents=True, exist_ok=True)
    events = json.loads(LOG.read_text(encoding="utf-8")) if LOG.exists() else []
    events.append({"at": utc(), **kw})
    LOG.write_text(json.dumps(events, indent=1), encoding="utf-8")


def run_seed(seed: int, data_root, results_root, rehearsal: bool, freeze_id, resume: bool) -> dict:
    paths = B.seed_paths(seed, data_root, results_root)
    prov_path = paths["provenance"]
    prov = json.loads(prov_path.read_text(encoding="utf-8")) if (resume and prov_path.exists()) else {
        "seed": seed, "freeze_id": freeze_id, "rehearsal": rehearsal, "git_commit": F.git("rev-parse", "HEAD"), "paths": {k: str(v) for k, v in paths.items()},
        "hashes": {}, "steps": {}}

    def save():
        B.write_json(prov_path, prov)

    def step(name, fn):
        if prov["steps"].get(name, {}).get("status") == "completed":
            return True
        t0 = time.time()
        rec = {"status": "running", "started": utc(), "resumed": bool(resume and name in prov["steps"])}
        prov["steps"][name] = rec
        save()
        try:
            rec["detail"] = fn()
            rec["status"] = "completed"
        except BaseException as e:  # noqa: BLE001 - recorded, never hidden; SystemExit from the generator guard is recorded too
            rec["status"], rec["error"], rec["traceback"] = "failed", f"{type(e).__name__}: {e}", traceback.format_exc()
        rec["finished"], rec["seconds"] = utc(), round(time.time() - t0, 1)
        save()
        log_event(seed=seed, step=name, status=rec["status"], error=rec.get("error"), resume=rec["resumed"])
        print(f"  seed {seed} [{name}] {rec['status']} ({rec['seconds']}s)" + (f": {rec['error']}" if rec["status"] == "failed" else ""), flush=True)
        return rec["status"] == "completed"

    def generate():
        if paths["decision_db"].exists():
            raise RuntimeError("decision.db already exists for this seed: refusing to overwrite or reuse it")
        rep = build(seed=seed, out_root=data_root, allow_heldout=not rehearsal)
        prov["hashes"].update({"decision_db": F.plain_sha256(paths["decision_db"]), "hidden_truth": F.plain_sha256(paths["hidden_truth"])})
        return {"generation_report_seed": rep["seed"], "row_counts": rep["row_counts"], "db_size_mb": rep["db_size_mb"], "generate_seconds": rep["generate_seconds"]}

    def backtest():
        if paths["forecast_store"].exists():
            raise RuntimeError("forecast store already exists for this seed: refusing to overwrite it")
        prov["hashes"]["decision_db_at_backtest_start"] = F.plain_sha256(paths["decision_db"])
        logs = run_backtest(paths["decision_db"], paths["forecast_store"], cfg=load_config())  # direct call: NOT scripts/run_backtest.py
        prov["hashes"]["forecast_store"] = F.plain_sha256(paths["forecast_store"])
        return {"reviews": len(logs), "issued_total": sum(x["issued"] for x in logs), "summaries_total": sum(x["summaries"] for x in logs),
                "withheld_total": {k: sum(x["withheld"].get(k, 0) for x in logs) for k in sorted({k for x in logs for k in x["withheld"]})}}

    def ties():
        problems = B.tie_checks(seed, paths, prov["hashes"], rehearsal=rehearsal)
        if problems:
            raise RuntimeError("artifact tie checks failed, results NOT computed: " + " | ".join(problems))
        return {"problems": [], "checks": "seed fields, hashes recorded at build time vs now, dev-artifact identity, store evidence/schedule hashes, content tie"}

    def evaluate():
        import sqlite3
        dec, st = sqlite3.connect(paths["decision_db"]), sqlite3.connect(paths["forecast_store"])
        try:
            hidden = json.loads(paths["hidden_truth"].read_text(encoding="utf-8"))
            res = B.evaluate_seed(seed, dec, st, hidden)
        finally:
            dec.close(), st.close()
        res["provenance_file"] = str(paths["provenance"].relative_to(ROOT)) if paths["provenance"].is_relative_to(ROOT) else str(paths["provenance"])
        res["artifact_hashes"] = dict(prov["hashes"])
        B.write_json(paths["results"], res)
        return {"results_file": str(paths["results"]), "scored_channel_reviews": res["policy_primary"]["overall"]["n_scored"]}

    for name, fn in (("generate", generate), ("backtest", backtest), ("tie_checks", ties), ("evaluate", evaluate)):
        if not step(name, fn):
            break
    prov["completed"] = all(prov["steps"].get(n, {}).get("status") == "completed" for n in ("generate", "backtest", "tie_checks", "evaluate"))
    save()
    return prov


def rehearse():
    """Development seed 1 through the same code path, into temporary folders. The result must reproduce evals/results/dev_eval.json."""
    global RES, LOG
    tmp = Path(tempfile.mkdtemp(prefix="rehearsal_"))
    out = ROOT / "evals" / "results" / "rehearsal"
    RES, LOG = out, out / "RUN_LOG.json"  # a rehearsal NEVER writes into the held-out results folder
    print(f"rehearsal on development seed 1 in {tmp}", flush=True)
    prov = run_seed(1, tmp / "data", out, rehearsal=True, freeze_id="REHEARSAL (not a freeze)", resume=False)
    if not prov["completed"]:
        print("REHEARSAL FAILED:", {k: v.get("error") for k, v in prov["steps"].items() if v["status"] == "failed"})
        return 1
    res = json.loads((out / "seed_1" / "baseline_eval.json").read_text(encoding="utf-8"))
    dev = json.loads((ROOT / "evals" / "results" / "dev_eval.json").read_text(encoding="utf-8"))
    cell = lambda r: next(x for x in r["forecasting"]["table"] if x["level"] == "cohort" and x["group"] == "overall")
    dev_hash = F.data_artifact_hashes(1)
    checks = {
        "policy_overall_equal": res["policy_primary"]["overall"] == dev["policy"]["overall"],
        "abstention_recall_equal": res["policy_primary"]["abstention"]["recall"] == dev["policy"]["abstention"]["recall"],
        "cohort_forecast_overall_equal": cell(res) == cell(dev),
        "all_forecasting_rows_equal": res["forecasting"]["table"] == dev["forecasting"]["table"],
        "scored_policy_rows_equal": res["policy_primary"]["rows"] == dev["policy"]["rows"],
        "per_scenario_equal": res["policy_primary"]["per_scenario"] == dev["policy"]["per_scenario"],
        "background_actions_equal": res["policy_primary"]["background_unscored_actions"] == dev["policy"]["background_unscored_actions"],
        "regenerated_decision_db_byte_identical_to_dev": prov["hashes"]["decision_db"] == dev_hash["decision_db"],
        "regenerated_hidden_truth_byte_identical_to_dev": prov["hashes"]["hidden_truth"] == dev_hash["hidden_truth"],
        "regenerated_forecast_store_byte_identical_to_dev": prov["hashes"]["forecast_store"] == dev_hash["forecast_store"],
    }
    summary = {"rehearsal_seed": 1, "all_reproduced": all(v for k, v in checks.items() if not k.startswith("regenerated_")), "checks": checks,
               "dev_numbers": {"policy_overall": res["policy_primary"]["overall"], "abstention_recall": res["policy_primary"]["abstention"]["recall"], "cohort_forecast_overall": cell(res)},
               "scenario_validity_dev_seed": {"n_pass": res["scenario_validity"]["n_pass"], "n_intents": res["scenario_validity"]["n_intents"],
                                              "scenarios_not_materialized": res["scenario_validity"]["scenarios_not_materialized"]},
               "steps": {k: v["status"] for k, v in prov["steps"].items()}}
    B.write_json(out / "rehearsal_summary.json", summary)
    print(json.dumps(summary, indent=1, default=str))
    return 0 if summary["all_reproduced"] else 1


def execute(resume: bool):
    problems = F.verify_freeze()
    if problems:
        print("REFUSING to run: the frozen state does not verify:\n  - " + "\n  - ".join(problems))
        return 2
    m = F.load_manifest()
    seeds = F.protocol()["seeds"]["registered"]
    if seeds != m["seeds"]["registered_held_out"]:
        print("REFUSING: the protocol's registered seeds differ from the manifest's")
        return 2
    RES.mkdir(parents=True, exist_ok=True)
    if resume:
        if not LOCK.exists() or json.loads(LOCK.read_text(encoding="utf-8"))["freeze_id"] != m["freeze_id"]:
            print("REFUSING --resume: no run lock for this freeze")
            return 2
        log_event(event="resume", freeze_id=m["freeze_id"])
    else:
        try:
            with open(LOCK, "x", encoding="utf-8") as f:
                json.dump({"freeze_id": m["freeze_id"], "started": utc(), "git_commit": F.git("rev-parse", "HEAD"), "seeds": seeds, "note": "the held-out baseline run is ONCE; delete nothing"}, f, indent=1)
        except FileExistsError:
            print("REFUSING: the held-out run has already been started (RUN_LOCK.json exists). It is run once; use --resume only to finish steps that did not complete.")
            return 2
        log_event(event="start", freeze_id=m["freeze_id"], seeds=seeds)
    for seed in seeds:
        print(f"seed {seed}", flush=True)
        run_seed(seed, ROOT / "data", RES, rehearsal=False, freeze_id=m["freeze_id"], resume=resume)
    log_event(event="finished")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--rehearse", action="store_true")
    g.add_argument("--execute", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    sys.exit(rehearse() if a.rehearse else execute(a.resume))
