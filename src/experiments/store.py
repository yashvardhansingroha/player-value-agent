"""SIMULATED experiment store (its own SQLite file). ALL DATA IS SYNTHETIC. It holds ONLY observable things: units and their budgets,
frozen plans, noisy observations, the simulation clock, run links and outcomes. Hidden truth never enters this database.

Database-level guards (each tested):
  * the simulation clock can only move forward;
  * a treatment-period observation can be inserted only if its plan was frozen (registered) first, on or before the insert date;
  * an observation can be inserted only once it is AVAILABLE (acquisition day + D30 horizon + reporting lag <= the clock); until
    then it does not exist, so a pending experiment has no outcome rows to read;
  * plans, observations, outcomes and events are append-only; at most one outcome per plan (no duplicate keep/rollback).
"""
import json
import sqlite3
from pathlib import Path

import pandas as pd

from src.experiments import plan as P

SCHEMA = """
CREATE TABLE sim_clock (id INTEGER PRIMARY KEY CHECK (id = 1), today TEXT NOT NULL);
CREATE TABLE units (unit_id TEXT PRIMARY KEY, channel TEXT NOT NULL, baseline_daily_budget REAL NOT NULL);
CREATE TABLE experiment_plans (plan_hash TEXT PRIMARY KEY, channel TEXT NOT NULL, plan_json TEXT NOT NULL, frozen_on TEXT NOT NULL);
CREATE TABLE observations (
    unit_id TEXT NOT NULL REFERENCES units (unit_id), acquisition_date TEXT NOT NULL,
    phase TEXT NOT NULL CHECK (phase IN ('pre', 'treatment')),
    spend REAL NOT NULL, installs INTEGER NOT NULL, d1_active INTEGER NOT NULL, d30_net_revenue REAL NOT NULL,
    available_on TEXT NOT NULL, generated_on TEXT NOT NULL, plan_hash TEXT,
    PRIMARY KEY (unit_id, acquisition_date)
);
CREATE TABLE experiment_runs (plan_hash TEXT PRIMARY KEY REFERENCES experiment_plans (plan_hash), change_id TEXT NOT NULL, applied_on TEXT NOT NULL);
CREATE TABLE experiment_outcomes (
    plan_hash TEXT PRIMARY KEY REFERENCES experiment_plans (plan_hash), outcome TEXT NOT NULL CHECK (outcome IN ('keep', 'rolled_back')),
    readout_json TEXT NOT NULL, ledger_result_json TEXT NOT NULL, executed_on TEXT NOT NULL
);
CREATE TABLE experiment_events (id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL, event TEXT NOT NULL, plan_hash TEXT, details_json TEXT NOT NULL);

CREATE TRIGGER clock_forward_only BEFORE UPDATE ON sim_clock WHEN NEW.today < OLD.today
    BEGIN SELECT RAISE(ABORT, 'the simulation clock cannot move backwards'); END;
CREATE TRIGGER obs_available_first BEFORE INSERT ON observations WHEN NEW.available_on > NEW.generated_on
    BEGIN SELECT RAISE(ABORT, 'observation is not available yet: pending outcomes cannot be stored'); END;
CREATE TRIGGER obs_plan_frozen_first BEFORE INSERT ON observations WHEN NEW.phase = 'treatment' AND (
        NEW.plan_hash IS NULL OR NOT EXISTS (SELECT 1 FROM experiment_plans p WHERE p.plan_hash = NEW.plan_hash AND p.frozen_on <= NEW.generated_on))
    BEGIN SELECT RAISE(ABORT, 'a treatment outcome needs a plan that was frozen first'); END;
"""
APPEND_ONLY = ("units", "experiment_plans", "observations", "experiment_runs", "experiment_outcomes", "experiment_events")


def create_store(path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path, isolation_level=None)
    conn.executescript(SCHEMA)
    for t in APPEND_ONLY:
        for op in ("UPDATE", "DELETE"):
            conn.execute(f"CREATE TRIGGER {t}_no_{op.lower()} BEFORE {op} ON {t} BEGIN SELECT RAISE(ABORT, '{t} is append-only'); END;")
    conn.execute("CREATE TRIGGER clock_no_delete BEFORE DELETE ON sim_clock BEGIN SELECT RAISE(ABORT, 'the clock cannot be deleted'); END;")
    return conn


def open_store(path) -> sqlite3.Connection:
    return sqlite3.connect(path, isolation_level=None)


def clock(conn) -> str:
    r = conn.execute("SELECT today FROM sim_clock WHERE id = 1").fetchone()
    if r is None:
        raise RuntimeError("simulation clock not set")
    return r[0]


def set_clock(conn, today: str) -> str:
    """First call sets the clock; later calls can only move it forward (database trigger)."""
    if conn.execute("SELECT COUNT(*) FROM sim_clock").fetchone()[0] == 0:
        conn.execute("INSERT INTO sim_clock (id, today) VALUES (1, ?)", (today,))
    else:
        conn.execute("UPDATE sim_clock SET today = ? WHERE id = 1", (today,))
    log_event(conn, "clock_set", None, {"today": today})
    return today


def log_event(conn, event: str, plan_hash, details: dict, at: str = None):
    has_clock = conn.execute("SELECT COUNT(*) FROM sim_clock").fetchone()[0] > 0
    when = at or (clock(conn) if has_clock else "n/a")
    conn.execute("INSERT INTO experiment_events (at, event, plan_hash, details_json) VALUES (?,?,?,?)", (when, event, plan_hash, json.dumps(details, sort_keys=True)))


def add_units(conn, channel: str, units: list):
    conn.executemany("INSERT INTO units VALUES (?,?,?)", [(u["unit_id"], channel, u["baseline_daily_budget"]) for u in units])


def units(conn, channel: str) -> list:
    return [{"unit_id": r[0], "baseline_daily_budget": r[1]} for r in
            conn.execute("SELECT unit_id, baseline_daily_budget FROM units WHERE channel = ? ORDER BY unit_id", (channel,))]


def register_plan(conn, plan: dict, experiment_cfg: dict) -> str:
    """Freeze a plan: store its canonical content under its recomputed hash, stamped with the current simulation date. The plan must be
    structurally valid AND follow the authoritative experiment configuration passed in (never one taken from the plan itself)."""
    errors = P.validate_plan_structure(plan)
    if errors:
        raise ValueError(f"plan is not structurally valid: {errors}")
    errors = P.validate_plan_against_config(plan, experiment_cfg)
    if errors:
        raise ValueError(f"plan does not follow the experiment configuration: {errors}")
    h = P.plan_hash(plan)
    from src.ledger.proposal import canonical
    conn.execute("INSERT INTO experiment_plans VALUES (?,?,?,?)", (h, plan["channel"], canonical(plan), clock(conn)))
    log_event(conn, "plan_frozen", h, {"channel": plan["channel"], "adequate": plan["power_assessment"]["adequate"]})
    return h


def get_plan(conn, plan_hash: str) -> dict:
    r = conn.execute("SELECT plan_json FROM experiment_plans WHERE plan_hash = ?", (plan_hash,)).fetchone()
    return None if r is None else json.loads(r[0])


def insert_observations(conn, rows: list):
    """Rows must already be available (acquisition day + horizon + lag <= clock); the database refuses anything else."""
    now = clock(conn)
    conn.executemany("INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?)",
                     [(r["unit_id"], r["acquisition_date"], r["phase"], r["spend"], r["installs"], r["d1_active"], r["d30_net_revenue"],
                       r["available_on"], now, r.get("plan_hash")) for r in rows])


def load_observations(conn, as_of_clock: str = None) -> pd.DataFrame:
    """Observed rows available at the clock (all stored rows are, by construction)."""
    c = as_of_clock or clock(conn)
    return pd.read_sql_query("SELECT * FROM observations WHERE available_on <= ?", conn, params=(c,))


def treatment_rows(conn, plan_hash: str) -> int:
    return conn.execute("SELECT COUNT(*) FROM observations WHERE plan_hash = ?", (plan_hash,)).fetchone()[0]
