"""THE single availability layer. Every decision-facing query reads the `v_*` views created here, never the base tables.

`set_as_of(conn, "2025-09-08")` means: the review happens at 00:00 of that date. A row is visible iff its
`reported_at <= as_of`. Players must also be visible for their days to count (an unattributed day cannot join a cohort).
The views are lazy and always recomputed from the base tables: nothing is cached.
"""
import sqlite3

VISIBLE_VIEWS = ("v_players", "v_player_days", "v_spend", "v_cohort_reporting")

_VIEW_SQL = """
CREATE TEMP TABLE IF NOT EXISTS _as_of (as_of TEXT NOT NULL);

CREATE TEMP VIEW IF NOT EXISTS v_players AS
    SELECT player_id, channel, install_date, reported_at
    FROM main.players
    WHERE reported_at <= (SELECT as_of FROM _as_of);

-- Joined 1:1 on the players primary key, so a player-day can never be duplicated.
CREATE TEMP VIEW IF NOT EXISTS v_player_days AS
    SELECT d.player_id, p.channel, p.install_date, d.event_date, d.day_n, d.sessions, d.levels,
           d.ad_impressions, d.iap_gross, d.ad_revenue, d.reported_at
    FROM main.player_days d
    JOIN main.players p ON p.player_id = d.player_id
    WHERE d.reported_at <= (SELECT as_of FROM _as_of)
      AND p.reported_at <= (SELECT as_of FROM _as_of);

CREATE TEMP VIEW IF NOT EXISTS v_spend AS
    SELECT channel, spend_date, spend, reported_at
    FROM main.spend_daily
    WHERE reported_at <= (SELECT as_of FROM _as_of);

CREATE TEMP VIEW IF NOT EXISTS v_cohort_reporting AS
    SELECT channel, install_date, through_day, complete_at
    FROM main.cohort_reporting
    WHERE complete_at <= (SELECT as_of FROM _as_of);
"""


def set_as_of(conn: sqlite3.Connection, as_of: str) -> None:
    """Create the visibility views (once per connection) and set the as-of date."""
    conn.executescript(_VIEW_SQL)
    conn.execute("DELETE FROM temp._as_of")
    conn.execute("INSERT INTO temp._as_of (as_of) VALUES (?)", (as_of,))
