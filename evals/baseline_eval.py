"""Seed-parameterised evaluation of the FROZEN DETERMINISTIC rules-only baseline (evals/heldout_protocol.yaml). ALL DATA IS SYNTHETIC.

It scores forecasts, policy decisions, abstentions and scenario validity on ONE seed's own artifacts, reusing the registered definitions in evals/dev_eval.py unchanged.
It uses no agent and no model call. Every artifact path contains the seed; the checks in `tie_checks` run BEFORE any result is computed, so a store built from one seed's data
can never be scored against another's.
The generator's hidden truth is read ONLY by `scenario_validity` (an evaluation-side diagnostic, never by any decision code).
"""
import json
import math
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals import dev_eval as E  # noqa: E402
from evals import freeze as F  # noqa: E402
from src.data.metrics import cohort_metrics  # noqa: E402
from src.forecast import pipeline, store  # noqa: E402
from src.generator.intents import compute_intent_metrics, evaluate_intents, load_scenarios  # noqa: E402
from src.ledger.proposal import jsonable  # noqa: E402
from src.quality import checks  # noqa: E402


def seed_paths(seed: int, data_root=None, results_root=None) -> dict:
    d, r = Path(data_root or ROOT / "data"), Path(results_root or ROOT / "evals" / "results" / "heldout")
    return {"decision_db": d / "decision" / f"seed_{seed}" / "decision.db", "generation_report": d / "decision" / f"seed_{seed}" / "generation_report.json",
            "hidden_truth": d / "hidden" / f"seed_{seed}" / "hidden_truth.json", "forecast_store": d / "forecasts" / f"seed_{seed}" / "forecasts.db",
            "results_dir": r / f"seed_{seed}", "results": r / f"seed_{seed}" / "baseline_eval.json", "provenance": r / f"seed_{seed}" / "provenance.json"}


# ------------------------------------------------------------------------------------------------ artifact ties


def _close(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)


def content_tie(dec, st, min_rows: int = 60, min_reviews: int = 8) -> dict:
    """Do the features stored in the forecast store equal the values recomputed from THIS seed's decision.db at the same review date?
    A store built from different data (another seed, a modified database) fails here."""
    cfg = pipeline.load_config()
    issued = pd.read_sql_query("SELECT channel, install_date, issued_as_of, features_json FROM forecast_issued ORDER BY issued_as_of, channel, install_date", st)
    reviews = sorted(issued.issued_as_of.unique())
    if len(reviews) < min_reviews:
        return {"passed": False, "reason": f"only {len(reviews)} reviews with issued forecasts (need {min_reviews})", "n_checked": 0, "n_reviews": len(reviews), "mismatches": []}
    per = max(3, math.ceil(min_rows / len(reviews)))
    checked, mismatches = 0, []
    for as_of in reviews:
        g = issued[issued.issued_as_of == as_of]
        k = min(per, len(g))
        take = g.iloc[sorted({int(i * len(g) / k) for i in range(k)})]
        m = cohort_metrics(dec, as_of)
        ann = checks.annotate_cohorts(m[m.channel != "organic"], cfg["quality"]).set_index(["channel", "install_date"])
        for r in take.itertuples(index=False):
            stored = json.loads(r.features_json)
            if (r.channel, r.install_date) not in ann.index:
                mismatches.append({"as_of": as_of, "channel": r.channel, "install_date": r.install_date, "problem": "cohort not present in this seed's data at that review date"})
                continue
            row = ann.loc[(r.channel, r.install_date)]
            bad = [k for k in pipeline.FEATURES if not _close(stored.get(k), pipeline._clean(row[k]))]
            checked += 1
            if bad:
                mismatches.append({"as_of": as_of, "channel": r.channel, "install_date": r.install_date, "differing_features": bad})
    ok = checked >= min_rows and not mismatches
    return {"passed": ok, "n_checked": checked, "n_reviews": len(reviews), "min_rows_required": min_rows, "mismatches": mismatches[:20], "n_mismatches": len(mismatches)}


def tie_checks(seed: int, paths: dict, recorded: dict, rehearsal: bool = False) -> list:
    """Return a list of problems (empty = every artifact is tied to this seed and unchanged). `recorded` holds the hashes written by the runner when each artifact was built.
    `rehearsal=True` (dev seed only, never the held-out run) skips the seed-registration and development-seed checks and the byte-identity check against the dev artifacts."""
    problems = []
    gen = yaml_gen()
    if not rehearsal and seed not in gen["heldout_seeds"]:
        problems.append(f"seed {seed} is not a registered held-out seed")
    if not rehearsal and seed in gen["dev_seeds"]:
        problems.append(f"seed {seed} is a development seed")
    if rehearsal and seed not in gen["dev_seeds"]:
        problems.append("a rehearsal may only use a development seed")
    for k in ("decision_db", "hidden_truth", "forecast_store", "generation_report"):
        if not Path(paths[k]).exists():
            problems.append(f"missing artifact: {k}")
    if problems:
        return problems
    hidden = json.loads(Path(paths["hidden_truth"]).read_text(encoding="utf-8"))
    rep = json.loads(Path(paths["generation_report"]).read_text(encoding="utf-8"))
    if hidden.get("seed") != seed:
        problems.append(f"hidden_truth.json states seed {hidden.get('seed')}, expected {seed}")
    if rep.get("seed") != seed:
        problems.append(f"generation_report.json states seed {rep.get('seed')}, expected {seed}")
    now = {"decision_db": F.plain_sha256(paths["decision_db"]), "hidden_truth": F.plain_sha256(paths["hidden_truth"]), "forecast_store": F.plain_sha256(paths["forecast_store"])}
    for k, h in now.items():
        if recorded.get(k) != h:
            problems.append(f"{k} hash at evaluation time differs from the hash recorded when it was built")
    if recorded.get("decision_db_at_backtest_start") != now["decision_db"]:
        problems.append("decision.db hash at the start of the backtest differs from its hash now (the store was not built from this exact database)")
    if not rehearsal:
        dev = F.data_artifact_hashes(gen["dev_seeds"][0])
        for k, h in now.items():
            if dev.get(k) == h:
                problems.append(f"{k} is byte-identical to the development seed's artifact")
    st = sqlite3.connect(paths["forecast_store"])
    dec = sqlite3.connect(paths["decision_db"])
    try:
        meta = store.read_meta(st)
        cfg = pipeline.load_config()
        if meta.get("forecast_evidence_hash") != store.forecast_evidence_hash(cfg):
            problems.append("forecast store evidence hash differs from the frozen forecasting configuration")
        if meta.get("schedule_hash") != store.sha256(cfg["backtest"]):
            problems.append("forecast store schedule hash differs from the frozen backtest schedule")
        tie = content_tie(dec, st)
        if not tie["passed"]:
            problems.append(f"content tie failed: {tie['n_mismatches'] if 'n_mismatches' in tie else 0} mismatching forecasts, {tie['n_checked']} checked ({tie.get('reason', '')})")
    finally:
        st.close()
        dec.close()
    return problems


def yaml_gen() -> dict:
    return E.GEN


# ------------------------------------------------------------------------------------------------ scenario validity


def scenario_validity(dec, hidden) -> dict:
    """Re-evaluate every committed generator calibration intent on THIS seed's data. A scenario with any FAIL is flagged `scenario_not_materialized` for this seed."""
    scenarios = load_scenarios()
    rows = evaluate_intents(compute_intent_metrics(dec, hidden), scenarios)
    table = [{"scenario": s, "metric": m, "value": float(v), "op": op, "threshold": thr, "pass": bool(ok)} for s, m, v, op, thr, ok in rows]
    flagged = sorted({r["scenario"] for r in table if not r["pass"]})
    return {"table": table, "n_intents": len(table), "n_pass": sum(r["pass"] for r in table), "scenarios_not_materialized": flagged,
            "note": "evaluation-side diagnostic using the generator's hidden truth; not a decision input. Flagged scenarios stay in the PRIMARY policy summary; the SECONDARY summary leaves them out."}


# ------------------------------------------------------------------------------------------------ one seed


def review_log(reviews: dict) -> list:
    return [{"date": dt, "channel": ch, "action": d["action"], "reasons": list(d["reasons"])} for dt, res in reviews["results"].items() for ch, d in res["decisions"].items()]


def evaluate_seed(seed: int, dec, st, hidden: dict) -> dict:
    forecasting = E.evaluate_forecasting(dec, st)
    reviews = E.run_reviews(dec, st)
    primary = E.evaluate_policy(reviews)
    validity = scenario_validity(dec, hidden)
    flagged = validity["scenarios_not_materialized"]
    secondary = None
    if flagged:
        try:
            secondary = E.evaluate_policy(reviews, exclude_scenarios=flagged)
        except Exception as e:  # noqa: BLE001 - recorded, never hidden; the primary summary stands
            secondary = {"error": f"{type(e).__name__}: {e}"}
    return {"seed": seed, "label": "HELD-OUT EVALUATION OF THE FROZEN DETERMINISTIC RULES-ONLY BASELINE on one additional synthetic seed. ALL DATA IS SYNTHETIC. It says nothing about any live agent.",
            "forecasting": forecasting, "policy_primary": primary, "policy_secondary_excluding_unmaterialized_scenarios": secondary, "scenario_validity": validity,
            "budgets": reviews["budgets"], "ceiling": reviews["ceiling"], "review_log": review_log(reviews)}


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(jsonable(obj), indent=1), encoding="utf-8")
