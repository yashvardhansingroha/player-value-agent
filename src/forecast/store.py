"""Forecast store: a SEPARATE SQLite file from the decision database. Issued forecasts, feature snapshots, withheld records
and weekly summaries are written once and never updated or deleted (database triggers enforce it). Nothing here
recomputes a forecast.

Note (docs/data_contract.md): storing what was forecast preserves the review's record; it does not make the underlying
decision database able to reconstruct history after a correction.
"""
import hashlib
import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE forecast_issued (
    channel TEXT NOT NULL,
    install_date TEXT NOT NULL,
    issued_as_of TEXT NOT NULL,
    method TEXT NOT NULL CHECK (method IN ('channel_multiplier', 'pooled_multiplier', 'zero_d3_fallback')),
    features_json TEXT NOT NULL,
    feature_hash TEXT NOT NULL,
    training_json TEXT NOT NULL,
    train_n INTEGER NOT NULL,
    train_max_outcome_complete_at TEXT NOT NULL,
    point_forecast REAL NOT NULL,
    reference_forecast REAL NOT NULL,
    observed_d3_floor REAL NOT NULL,            -- D0-D3 net revenue already observed: cumulative D30 cannot be lower
    point_floor_adjusted INTEGER NOT NULL,      -- 1 if the point forecast was raised to the floor
    interval_status TEXT NOT NULL CHECK (interval_status IN ('available', 'unavailable')),
    interval_lo REAL,
    interval_hi REAL,
    interval_floor_adjusted INTEGER NOT NULL,   -- 1 if lo and/or hi were raised to the floor (lo <= hi is preserved)
    interval_reason TEXT,
    calib_n INTEGER NOT NULL,
    calib_max_outcome_complete_at TEXT,
    config_hash TEXT NOT NULL,
    PRIMARY KEY (channel, install_date)
);
CREATE TABLE forecast_withheld (
    channel TEXT NOT NULL,
    install_date TEXT NOT NULL,
    attempted_as_of TEXT NOT NULL,
    reason TEXT NOT NULL,
    detail_json TEXT NOT NULL,
    PRIMARY KEY (channel, install_date)
);
CREATE TABLE weekly_summary (
    channel TEXT NOT NULL,
    as_of TEXT NOT NULL,
    window_start TEXT,
    window_end TEXT,
    included_json TEXT NOT NULL,
    withheld_json TEXT NOT NULL,
    accounting_json TEXT NOT NULL,              -- EVERY expected install date: forecast | withheld | pending_incomplete | missing | unaccounted
    fully_decision_ready INTEGER NOT NULL,      -- 0 if any expected date is pending, missing or unaccounted, or completeness is unexpected
    not_ready_reasons_json TEXT NOT NULL,
    n_expected_dates INTEGER NOT NULL,
    n_included INTEGER NOT NULL,
    n_withheld INTEGER NOT NULL,
    installs_included INTEGER NOT NULL,
    spend_included REAL NOT NULL,
    point_forecast REAL,
    observed_d3_floor REAL,                     -- sum of D0-D3 net revenue of the included cohorts
    interval_status TEXT NOT NULL CHECK (interval_status IN ('available', 'unavailable')),
    interval_lo REAL,
    interval_hi REAL,
    interval_floor_adjusted INTEGER NOT NULL,
    interval_reason TEXT,
    calib_n INTEGER NOT NULL,
    quality_json TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    PRIMARY KEY (channel, as_of)
);
"""
META_SCHEMA = "CREATE TABLE store_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
TABLES = ("forecast_issued", "forecast_withheld", "weekly_summary", "store_meta")


def create_store(path, schedule: dict = None, evidence_hash: str = None) -> sqlite3.Connection:
    """Create an empty store. `schedule` (the run-scheduling settings) and `evidence_hash` are recorded SEPARATELY in store_meta."""
    path = Path(path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA + META_SCHEMA)
    for t in TABLES:
        for op in ("UPDATE", "DELETE"):
            conn.execute(f"CREATE TRIGGER {t}_no_{op.lower()} BEFORE {op} ON {t} BEGIN SELECT RAISE(ABORT, '{t} is immutable'); END;")
    meta = {}
    if evidence_hash is not None:
        meta["forecast_evidence_hash"] = evidence_hash
    if schedule is not None:
        meta["schedule_json"] = canonical_json(schedule)
        meta["schedule_hash"] = sha256(schedule)
    conn.executemany("INSERT INTO store_meta (key, value) VALUES (?, ?)", sorted(meta.items()))
    conn.commit()
    return conn


def read_meta(conn: sqlite3.Connection) -> dict:
    return dict(conn.execute("SELECT key, value FROM store_meta").fetchall())


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def sha256(obj) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def forecast_evidence_hash(cfg: dict) -> str:
    """Hash of the FORECASTING AND QUALITY settings: the `quality` and `forecast` sections of config/forecast.yaml plus its version.
    Every issued forecast and weekly summary stores this hash; a policy decision refuses evidence whose stored hash differs from the
    hash of the configuration it is run under.

    SCOPE, stated precisely: the hash does NOT cover run scheduling (the `backtest` section: first review, last review, review
    frequency, scoring date). That is not because scheduling is irrelevant: changing the first review or the review frequency can
    change which cohorts are issued, when, and the calibration history behind every interval. Scheduling is therefore recorded
    SEPARATELY, as `schedule_json` / `schedule_hash` in the `store_meta` table, and surfaced in policy evidence, so a reader can see
    which schedule produced the store. The hash only says the forecasting and quality settings match."""
    return sha256({"version": cfg.get("version"), "quality": cfg["quality"], "forecast": cfg["forecast"]})


def attempted_keys(conn: sqlite3.Connection) -> set:
    rows = conn.execute("SELECT channel, install_date FROM forecast_issued UNION SELECT channel, install_date FROM forecast_withheld").fetchall()
    return set(rows)
