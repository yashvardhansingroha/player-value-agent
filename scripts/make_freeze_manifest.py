"""Write evals/FREEZE_MANIFEST.json for the deterministic-baseline held-out evaluation. ALL DATA IS SYNTHETIC.

    python scripts/make_freeze_manifest.py

Preconditions (each refused otherwise): the dev-seed rehearsal reproduced the development numbers; no held-out data and no run lock exist; the manifest does not already exist.
After writing it, COMMIT everything frozen together with the manifest, then create the annotated tag named in `freeze_id`:
    git tag -a baseline-freeze-v1 -m "..."
The runner verifies the working tree against that tag before it generates any held-out data.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals import freeze as F  # noqa: E402


def main() -> int:
    if F.MANIFEST.exists():
        print("REFUSING: the manifest already exists; a freeze is not rewritten")
        return 2
    held = [p for s in F.protocol()["seeds"]["registered"] for p in (ROOT / "data").glob(f"*/seed_{s}")]
    if held or (ROOT / "evals" / "results" / "heldout").exists():
        print("REFUSING: held-out artifacts or a held-out results folder already exist:", [str(p) for p in held])
        return 2
    rp = ROOT / "evals" / "results" / "rehearsal" / "rehearsal_summary.json"
    if not rp.exists():
        print("REFUSING: run `python scripts/run_heldout_baseline.py --rehearse` first")
        return 2
    reh = json.loads(rp.read_text(encoding="utf-8"))
    if not reh["all_reproduced"]:
        print("REFUSING: the rehearsal did not reproduce the development numbers")
        return 2
    m = F.build_manifest({"summary_file": "evals/results/rehearsal/rehearsal_summary.json", "all_reproduced": True, "checks": reh["checks"], "dev_numbers": reh["dev_numbers"]})
    F.MANIFEST.write_text(json.dumps(m, indent=1), encoding="utf-8")
    print(f"wrote {F.MANIFEST.relative_to(ROOT)}: {len(m['frozen_files'])} frozen files, parent commit {m['parent_commit'][:10]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
