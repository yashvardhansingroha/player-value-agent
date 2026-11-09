"""Create the decision SQLite database from generated tables. Hidden truth is not accepted here on purpose."""
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABLES = ("players", "player_days", "spend_daily", "cohort_reporting")


def create_database(path, players, player_days, spend_daily, cohort_reporting, meta: dict) -> dict:
    """Write the four decision tables + meta. Returns row counts. Replaces any existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    try:
        conn.executescript((ROOT / "sql" / "schema.sql").read_text(encoding="utf-8"))
        conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", list(meta.items()))
        players.to_sql("players", conn, if_exists="append", index=False, chunksize=50_000)
        player_days.to_sql("player_days", conn, if_exists="append", index=False, chunksize=50_000)
        spend_daily.to_sql("spend_daily", conn, if_exists="append", index=False, chunksize=50_000)
        cohort_reporting.to_sql("cohort_reporting", conn, if_exists="append", index=False, chunksize=50_000)
        conn.commit()
        counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}
    finally:
        conn.close()
    return counts
