"""Print whether the generated dev data satisfies the scenario calibration intents (generator QA, not a decision).

    python scripts/calibration_report.py [--seed 1]
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generator.intents import compute_intent_metrics, evaluate_intents, load_scenarios  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    conn = sqlite3.connect(ROOT / "data" / "decision" / f"seed_{args.seed}" / "decision.db")
    hidden = json.loads((ROOT / "data" / "hidden" / f"seed_{args.seed}" / "hidden_truth.json").read_text(encoding="utf-8"))
    metrics = compute_intent_metrics(conn, hidden)
    print("Per paid channel, cohorts installed on days 0..184 (D180 ROAS = avg_roas / target):")
    for (sid, ch), s in metrics.items():
        if sid == "_summary":
            print(f"  {ch:11s} avg D30 ROAS {s['avg_roas']:.2f}  target D30 ROAS {s['target']:.2f}  D180/D30 {1 / s['target']:.2f}  "
                  f"D30/D3 {s['mult']:.2f}  D3 net/install {s['d3_per_install']:.3f}  D30 net/install {s['v_ref']:.3f}")
    print()
    rows = evaluate_intents(metrics, load_scenarios())
    for sid, metric, value, op, thr, ok in rows:
        print(f"{'PASS' if ok else 'FAIL'}  {sid:28s} {metric:42s} {value:9.3f} {op} {thr}")
    print(f"\n{sum(r[-1] for r in rows)}/{len(rows)} generator-QA checks satisfied (generator was tuned on the dev seed; "
          "true_step20_* is an approximate calibration proxy, not simulation truth)")
