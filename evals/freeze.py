"""Freeze manifest for the DETERMINISTIC-BASELINE held-out evaluation (evals/heldout_protocol.yaml). ALL DATA IS SYNTHETIC.

The manifest ties together, by sha256 and by an annotated git tag:
  * every frozen source / configuration / evaluation-definition file,
  * the agent system prompt and tool declarations (frozen although this run does not use them),
  * the seed lists and the development artifacts (so development data cannot be passed off as held-out data),
  * the numbers the harness must reproduce on the development seed (the rehearsal).
`verify_freeze` is called by the held-out runner BEFORE any held-out data is generated. Text files are hashed with CRLF normalised to LF so the hash does not depend on
the checkout's line-ending setting.
"""
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "evals" / "FREEZE_MANIFEST.json"
PROTOCOL = ROOT / "evals" / "heldout_protocol.yaml"
FREEZE_ID = "baseline-freeze-v1"

FROZEN_GLOBS = [
    "src/**/*.py", "config/*.yaml", "sql/*.sql", "requirements.txt", "requirements.in",
    "evals/__init__.py", "evals/criteria.yaml", "evals/heldout_protocol.yaml", "evals/scenarios/*", "evals/dev_eval.py", "evals/baseline_eval.py", "evals/freeze.py",
    "scripts/__init__.py", "scripts/generate_data.py", "scripts/run_heldout_baseline.py", "scripts/make_freeze_manifest.py",
]
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".md", ".txt", ".sql", ".in"}


def file_hash(path: Path) -> str:
    b = Path(path).read_bytes()
    if Path(path).suffix in TEXT_SUFFIXES:
        b = b.replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()


def plain_sha256(path: Path) -> str:
    """Binary hash for data files (databases, JSON truth) - no normalisation."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_files() -> list:
    out = set()
    for g in FROZEN_GLOBS:
        for p in ROOT.glob(g):
            if p.is_file() and "__pycache__" not in p.parts:
                out.add(p.relative_to(ROOT).as_posix())
    return sorted(out)


def protocol() -> dict:
    return yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))


def agent_surface_hashes() -> dict:
    """The agent prompt and tool declarations: frozen so a later agent evaluation starts from a recorded state. They are NOT used by this run."""
    sys.path.insert(0, str(ROOT))
    from src.agent import agent as A
    from src.agent import tools as T
    h = lambda b: hashlib.sha256(b).hexdigest()
    return {"system_prompt_sha256": h(A.SYSTEM_PROMPT.encode()), "tool_declarations_sha256": h(json.dumps(T.declarations(), sort_keys=True).encode())}


def data_artifact_hashes(seed: int, root: Path = None) -> dict:
    root = Path(root or ROOT / "data")
    paths = {"decision_db": root / "decision" / f"seed_{seed}" / "decision.db", "hidden_truth": root / "hidden" / f"seed_{seed}" / "hidden_truth.json",
             "forecast_store": root / "forecasts" / f"seed_{seed}" / "forecasts.db"}
    return {k: (plain_sha256(p) if p.exists() else None) for k, p in paths.items()}


def git(*args, check=True) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def environment() -> dict:
    import numpy, pandas, scipy
    return {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": numpy.__version__, "pandas": pandas.__version__, "scipy": scipy.__version__,
            "note": "informational: the generator and the deterministic pipeline are seeded; a different library version could change floating-point results"}


def build_manifest(dev_rehearsal: dict) -> dict:
    gen = yaml.safe_load((ROOT / "config" / "generator.yaml").read_text(encoding="utf-8"))
    proto = protocol()
    return {
        "freeze_id": FREEZE_ID,
        "label": "FREEZE OF THE DETERMINISTIC RULES-ONLY BASELINE. ALL DATA IS SYNTHETIC. It authorises no claim about any live agent.",
        "scope_amendment": "evals/criteria.yaml amendment dated 2026-11-08",
        "parent_commit": git("rev-parse", "HEAD"),
        "note_on_commit": "This manifest is committed together with the frozen files; the annotated tag named in freeze_id points at that commit. verify_freeze checks the working tree against the tag.",
        "seeds": {"registered_held_out": gen["heldout_seeds"], "development": gen["dev_seeds"], "protocol_registered": proto["seeds"]["registered"]},
        "frozen_files": {p: file_hash(ROOT / p) for p in frozen_files()},
        "agent_surface": agent_surface_hashes(),
        "development_artifacts_seed_1": data_artifact_hashes(gen["dev_seeds"][0]),
        "dev_rehearsal_reproduces": dev_rehearsal,
        "environment": environment(),
    }


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def verify_freeze(manifest: dict = None, check_tag: bool = True, check_dev_artifacts: bool = True) -> list:
    """Return a list of problems (empty = frozen state verified)."""
    problems = []
    if not MANIFEST.exists() and manifest is None:
        return ["evals/FREEZE_MANIFEST.json does not exist: nothing has been frozen"]
    m = manifest or load_manifest()
    for p, h in m["frozen_files"].items():
        f = ROOT / p
        if not f.exists():
            problems.append(f"frozen file missing: {p}")
        elif file_hash(f) != h:
            problems.append(f"frozen file changed since the freeze: {p}")
    extra = set(frozen_files()) - set(m["frozen_files"])
    problems += [f"file matches a frozen pattern but is not in the manifest: {p}" for p in sorted(extra)]
    try:
        cur = agent_surface_hashes()
        problems += [f"agent surface changed: {k}" for k in cur if cur[k] != m["agent_surface"][k]]
    except Exception as e:  # noqa: BLE001
        problems.append(f"agent surface could not be hashed: {e}")
    if check_dev_artifacts:
        cur = data_artifact_hashes(m["seeds"]["development"][0])
        problems += [f"development artifact changed or missing: {k}" for k, v in m["development_artifacts_seed_1"].items() if cur.get(k) != v]
    if check_tag:
        try:
            if git("cat-file", "-t", m["freeze_id"], check=False) != "tag":
                problems.append(f"git tag {m['freeze_id']} is missing or not an annotated tag")
            else:
                for p in m["frozen_files"]:
                    blob = git("rev-parse", f"{m['freeze_id']}^{{commit}}:{p}", check=False)
                    here = git("hash-object", p, check=False)
                    if not blob or blob != here:
                        problems.append(f"working-tree file differs from the tagged commit: {p}")
                tagged_manifest = git("show", f"{m['freeze_id']}:evals/FREEZE_MANIFEST.json", check=False)
                if not tagged_manifest or json.loads(tagged_manifest) != m:
                    problems.append("evals/FREEZE_MANIFEST.json differs from the copy in the tagged commit")
        except Exception as e:  # noqa: BLE001
            problems.append(f"git tag check failed: {e}")
    return problems
