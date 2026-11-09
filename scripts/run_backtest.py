"""Run the chronological forecast backtest on one dataset and write the forecast store + scores.

    python scripts/run_backtest.py [--seed 1]

Reads data/decision/seed_<n>/decision.db (decision data only). Writes data/forecasts/seed_<n>/forecasts.db and
reports/phase2_backtest_results.json. Config comes from config/forecast.yaml; see the provenance note at the top of that file
for which settings were fixed in advance and which detector settings changed after development inspection.
Progress is printed for both the main run and the labelled sensitivity run.
"""
import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.calendar import day_to_date  # noqa: E402
from src.forecast import scoring  # noqa: E402
from src.forecast.pipeline import START, load_config, run_backtest  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    cfg = load_config()
    dec_path = ROOT / "data" / "decision" / f"seed_{args.seed}" / "decision.db"
    store_path = ROOT / "data" / "forecasts" / f"seed_{args.seed}" / "forecasts.db"
    t0 = time.time()
    n_reviews = len(range(cfg["backtest"]["first_as_of_day"], cfg["backtest"]["last_as_of_day"] + 1, cfg["backtest"]["review_every_days"]))

    def progress(label):
        done = [0]

        def show(l):
            done[0] += 1
            print(f"[{label}] review {done[0]:2d}/{n_reviews} day {l['as_of_day']:3d} ({l['as_of']}): issued {l['issued']:3d}, "
                  f"withheld {sum(l['withheld'].values()):3d}, summaries {l['summaries']}  [{time.time() - t0:5.0f}s elapsed]", flush=True)
        return show

    print("[main] starting main backtest", flush=True)
    logs = run_backtest(dec_path, store_path, cfg, progress=progress("main"))
    elapsed = time.time() - t0
    print(f"[main] finished in {elapsed:.0f}s", flush=True)
    scoring_as_of = day_to_date(START, cfg["backtest"]["scoring_as_of_day"])
    dec, st = sqlite3.connect(dec_path), sqlite3.connect(store_path)
    results = {
        "seed": args.seed,
        "note": "DEV SEED backtest (generator was calibrated on this seed). Not held-out evidence. ALL DATA IS SYNTHETIC.",
        "config": cfg,
        "backtest_seconds": round(elapsed, 1),
        "reviews": [{k: v for k, v in l.items()} for l in logs],
        "cohort_forecasts": scoring.score_cohorts(st, dec, scoring_as_of),
        "weekly_summaries": scoring.score_summaries(st, dec, scoring_as_of),
    }
    # Labelled SENSITIVITY run (not the chosen configuration): same pipeline with the revenue-spike exclusion switched off,
    # scored on the cohorts the main run withheld for a spike, so the cost of that exclusion is visible.
    import copy

    import pandas as pd

    cfg2 = copy.deepcopy(cfg)
    cfg2["quality"]["spike_ratio"] = 1e12
    store2 = ROOT / "data" / "forecasts" / f"seed_{args.seed}" / "forecasts_no_spike_exclusion.db"
    print("[sensitivity] starting sensitivity run (revenue-spike exclusion OFF; not the chosen configuration)", flush=True)
    run_backtest(dec_path, store2, cfg2, progress=progress("sensitivity"))
    print(f"[sensitivity] finished; total {time.time() - t0:.0f}s", flush=True)
    st2 = sqlite3.connect(store2)
    spike_keys = set(map(tuple, pd.read_sql_query("SELECT channel, install_date FROM forecast_withheld WHERE reason = 'revenue_spike'", st).to_numpy()))
    results["sensitivity_no_spike_exclusion"] = {
        "note": "NOT the chosen configuration. Revenue-spike exclusion disabled (spike_ratio = 1e12); training also keeps those cohorts.",
        "all_cohorts": scoring.score_cohorts(st2, dec, scoring_as_of)["overall"],
        "on_cohorts_the_main_run_withheld_for_spike": scoring.score_subset(st2, dec, scoring_as_of, spike_keys),
        "on_cohorts_the_main_run_issued": scoring.score_subset(
            st2, dec, scoring_as_of,
            set(map(tuple, pd.read_sql_query("SELECT channel, install_date FROM forecast_issued", st).to_numpy()))),
    }
    out = ROOT / "reports" / "phase2_backtest_results.json"
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nbacktest {elapsed:.0f}s; wrote {out}")
