"""Check this repository's files against the recorded baseline freeze (evals/FREEZE_MANIFEST.json). ALL DATA IS SYNTHETIC.

The manifest records the sha256 of 78 files as they were when the deterministic baseline was frozen and run on the held-out seeds. This repository has a fresh git history, so the
annotated tag named in the manifest is not available here and is not checked. What CAN be checked is the file content:

  * a frozen file that is unchanged still matches its recorded hash;
  * every difference from the freeze is declared in evals/publication_delta.json with the current hash of the file and the reason
    (the Groq provider adapter added after the freeze, and one wording edit). Anything undeclared, or a declared file whose content has changed again, is reported.

The manifest, the held-out protocol results and the run lock are kept as recorded and are never rewritten.

    python scripts/verify_baseline_provenance.py        # exit 0 and "ok" when everything is accounted for
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals import freeze as F  # noqa: E402

DELTA = ROOT / "evals" / "publication_delta.json"


def check(check_dev_artifacts: bool = False) -> list:
    """Problems (empty = every difference from the recorded freeze is declared and current)."""
    delta = json.loads(DELTA.read_text(encoding="utf-8"))["files"]
    problems, seen = [], set()
    for line in F.verify_freeze(check_tag=False, check_dev_artifacts=check_dev_artifacts):
        m = re.search(r": (\S+)$", line)
        path = m.group(1) if m else None
        if line.startswith(("frozen file changed", "file matches a frozen pattern")) and path in delta:
            seen.add(path)
            if F.file_hash(ROOT / path) != delta[path]["sha256"]:
                problems.append(f"declared file changed again after it was declared: {path}")
        else:
            problems.append(line)
    problems += [f"declared in publication_delta.json but identical to the freeze (stale entry): {p}" for p in sorted(set(delta) - seen)]
    return problems


if __name__ == "__main__":
    bad = check(check_dev_artifacts="--dev-artifacts" in sys.argv)
    print("\n".join(bad) if bad else "ok: every difference from the recorded freeze is declared and current")
    sys.exit(1 if bad else 0)
