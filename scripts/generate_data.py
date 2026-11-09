"""Generate one synthetic dataset.

    python scripts/generate_data.py                 # dev seed from config (seed 1)
    python scripts/generate_data.py --seed 2

Writes two SEPARATE trees:
    data/decision/seed_<n>/decision.db        decision-facing tables only (the agent/tools may read this)
    data/hidden/seed_<n>/hidden_truth.json    hidden simulation truth (never read by decision code)
    data/decision/seed_<n>/generation_report.json   runtime and row counts
Held-out seeds from config are refused unless --allow-heldout (reserved for the final evaluation run).
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.load import create_database  # noqa: E402
from src.generator.generate import generate, load_config  # noqa: E402


def build(seed=None, out_root=None, install_scale=1.0, cfg=None, allow_heldout=False):
    cfg = cfg or load_config()
    seed = cfg["seed"] if seed is None else seed
    if seed in cfg["heldout_seeds"] and not allow_heldout:
        raise SystemExit(f"seed {seed} is a held-out seed; refusing (use --allow-heldout only for the final evaluation).")
    out_root = Path(out_root or ROOT / "data")
    decision_dir = out_root / "decision" / f"seed_{seed}"
    hidden_dir = out_root / "hidden" / f"seed_{seed}"
    hidden_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    data = generate(cfg, seed=seed, install_scale=install_scale)
    t_generate = time.time() - t0
    t1 = time.time()
    db_path = decision_dir / "decision.db"
    counts = create_database(db_path, data.players, data.player_days, data.spend_daily, data.cohort_reporting, data.meta)
    t_load = time.time() - t1

    (hidden_dir / "hidden_truth.json").write_text(json.dumps(data.hidden_truth), encoding="utf-8")
    report = {
        "seed": seed,
        "install_scale": install_scale,
        "generate_seconds": round(t_generate, 1),
        "load_seconds": round(t_load, 1),
        "row_counts": counts,
        "db_size_mb": round(db_path.stat().st_size / 1e6, 1),
        "db_path": str(db_path),
        "hidden_truth_path": str(hidden_dir / "hidden_truth.json"),
    }
    (decision_dir / "generation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--allow-heldout", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(seed=args.seed, allow_heldout=args.allow_heldout), indent=2))
