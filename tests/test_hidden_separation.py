"""Hidden simulation truth must stay out of the decision database and out of decision-facing code.
These are selected checks, not a proof that no leakage path exists."""
import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_TABLES = {"meta", "players", "player_days", "spend_daily", "cohort_reporting"}
ALLOWED_COLUMNS = {
    "meta": {"key", "value"},
    "players": {"player_id", "channel", "install_date", "reported_at"},
    "player_days": {"player_id", "event_date", "day_n", "sessions", "levels", "ad_impressions", "iap_gross", "ad_revenue", "reported_at"},
    "spend_daily": {"channel", "spend_date", "spend", "reported_at"},
    "cohort_reporting": {"channel", "install_date", "through_day", "complete_at"},
}
ALLOWED_META_KEYS = {"schema_version", "start_date", "horizon_days", "last_event_date", "max_activity_day_n", "store_fee", "synthetic"}
# Decision-facing code. Exactly TWO experiment files are HIDDEN-SIDE and may read hidden truth / simulator parameters: the simulator that
# generates noisy observations from the truth, and the scoring that compares an estimate with the truth AFTER the decision.
# Everything else in src/experiments (plan, design, readout, store, lifecycle) is decision-facing and is scanned like the rest.
HIDDEN_SIDE_FILES = {"src/experiments/simulator.py", "src/experiments/scoring.py"}
DECISION_DIRS = ["src/data", "src/quality", "src/forecast", "src/policy", "src/experiments", "src/agent", "src/validate", "src/ledger", "app"]
FORBIDDEN_IN_CODE = ["src.generator", "from src import generator", "hidden_truth", "evals/scenarios", "evals\\\\scenarios", "data/hidden"]


def test_decision_db_has_only_decision_tables_and_columns(conn):
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert tables == ALLOWED_TABLES
    for t, cols in ALLOWED_COLUMNS.items():
        assert {r[1] for r in conn.execute(f"PRAGMA table_info({t})")} == cols
    assert {r[0] for r in conn.execute("SELECT key FROM meta")} == ALLOWED_META_KEYS


def test_hidden_truth_is_in_a_separate_tree_and_not_inside_the_database(dataset):
    db, hidden = dataset["db"], dataset["hidden"]
    assert hidden.exists() and "hidden" in hidden.parts and "decision" in db.parts
    assert db.parent not in hidden.parents  # hidden file is not under the decision folder
    blob = db.read_bytes()
    for needle in (b"quality_gamma", b"daily_mean_installs_noiseless", b"event_loss_probability", b"whale", b"engagement"):
        assert needle not in blob, needle


def test_decision_facing_code_never_references_hidden_truth_or_scenario_files():
    offenders = []
    for d in DECISION_DIRS:
        for path in (ROOT / d).rglob("*.py"):
            if path.relative_to(ROOT).as_posix() in HIDDEN_SIDE_FILES:
                continue
            text = path.read_text(encoding="utf-8")
            offenders += [f"{path.relative_to(ROOT)}: {s}" for s in FORBIDDEN_IN_CODE if s in text]
    assert not offenders, offenders


def test_only_the_two_hidden_side_experiment_files_read_hidden_truth():
    """Across ALL of src/, `hidden_truth` appears only in the generator and the two allow-listed hidden-side files."""
    mentions = {p.relative_to(ROOT).as_posix() for p in (ROOT / "src").rglob("*.py") if "hidden_truth" in p.read_text(encoding="utf-8")}
    generator = {m for m in mentions if m.startswith("src/generator/")}
    assert mentions - generator == {"src/experiments/simulator.py"}  # the simulator loads it; scoring receives the simulator object
    for name in HIDDEN_SIDE_FILES:
        assert (ROOT / name).exists()


def test_metric_sql_reads_only_availability_views():
    allowed = {"v_players", "v_player_days", "v_spend", "v_cohort_reporting", "meta"}
    for path in (ROOT / "sql").glob("*.sql"):
        if path.name == "schema.sql":
            continue
        text = re.sub(r"--[^\n]*", "", path.read_text(encoding="utf-8"))
        ctes = set(re.findall(r"(\w+)\s+AS\s*\(", text, flags=re.IGNORECASE))
        sources = set(re.findall(r"\b(?:FROM|JOIN)\s+(\w+)", text, flags=re.IGNORECASE))
        assert sources <= allowed | ctes, f"{path.name}: {sources - allowed - ctes}"
        for base in ("players", "player_days", "spend_daily", "cohort_reporting"):
            assert not re.search(rf"\b{base}\b", text), f"{path.name} reads base table {base}"
