"""Cohort metric functions. Each takes an explicit `as_of` date and recomputes from the database every call
(no caching). SQL in sql/*.sql is the source of truth; these functions only bind parameters."""
import sqlite3
from pathlib import Path

import pandas as pd

from src.data.availability import set_as_of

ROOT = Path(__file__).resolve().parents[2]
MILESTONES = (3, 30, 180)


def _sql(name: str) -> str:
    return (ROOT / "sql" / f"{name}.sql").read_text(encoding="utf-8")


def store_fee(conn: sqlite3.Connection) -> float:
    return float(conn.execute("SELECT value FROM meta WHERE key = 'store_fee'").fetchone()[0])


def cohort_metrics(conn: sqlite3.Connection, as_of: str) -> pd.DataFrame:
    """One row per cohort (channel x install_date) known at `as_of`. D3 metrics are NULL until D3 is complete."""
    set_as_of(conn, as_of)
    return pd.read_sql_query(_sql("cohort_metrics"), conn, params={"fee": store_fee(conn)})


def cohort_outcomes(conn: sqlite3.Connection, as_of: str, through_day: int) -> pd.DataFrame:
    """Cumulative D0..D<through_day> net revenue for cohorts whose milestone is complete at `as_of`."""
    if through_day not in MILESTONES:
        raise ValueError(f"through_day must be one of {MILESTONES}")
    set_as_of(conn, as_of)
    return pd.read_sql_query(_sql("cohort_outcomes"), conn, params={"through_day": through_day, "fee": store_fee(conn)})


def milestone_complete_at(conn: sqlite3.Connection, as_of: str, through_day: int) -> pd.DataFrame:
    """Declared completeness date of the D<through_day> milestone, for cohorts already complete at `as_of`."""
    if through_day not in MILESTONES:
        raise ValueError(f"through_day must be one of {MILESTONES}")
    set_as_of(conn, as_of)
    return pd.read_sql_query(_sql("milestone_complete_at"), conn, params={"through_day": through_day})


def daily_activity(conn: sqlite3.Connection, as_of: str, channel: str) -> pd.DataFrame:
    """Visible active player-days per spend date, with players installed in the previous 30 days."""
    set_as_of(conn, as_of)
    return pd.read_sql_query(_sql("daily_activity"), conn, params={"channel": channel})


def payer_gross(conn: sqlite3.Connection, as_of: str, channel: str, from_date: str, to_date: str) -> pd.DataFrame:
    """D0-D3 gross IAP per paying player for cohorts installed in [from_date, to_date]."""
    set_as_of(conn, as_of)
    return pd.read_sql_query(_sql("payer_concentration"), conn,
                             params={"channel": channel, "from_date": from_date, "to_date": to_date})


def retention_curve(conn: sqlite3.Connection, as_of: str, channel: str) -> pd.DataFrame:
    """Classic retention D0..D30 (always 31 rows) for one channel over cohorts matured for D30 at `as_of`.
    Days with no activity have 0 active players and 0 retention. When no eligible installed players exist,
    `retention` is NaN and `retention_available` is 0: unavailable, not zero."""
    set_as_of(conn, as_of)
    return pd.read_sql_query(_sql("retention_curve"), conn, params={"channel": channel})
