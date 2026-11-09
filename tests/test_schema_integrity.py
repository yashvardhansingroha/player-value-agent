"""Grain, keys and reporting-metadata consistency on the generated database."""
import pandas as pd

PAID = ["meta", "google_uac", "applovin", "unity", "tiktok"]


def one(conn, sql, *args):
    return conn.execute(sql, args).fetchone()[0]


def test_row_counts_and_grains(conn, dataset):
    cfg = dataset["cfg"]
    assert one(conn, "SELECT COUNT(*) FROM spend_daily") == len(PAID) * cfg["horizon_days"]
    assert one(conn, "SELECT COUNT(*) FROM (SELECT channel, spend_date FROM spend_daily GROUP BY 1,2 HAVING COUNT(*) > 1)") == 0
    assert one(conn, "SELECT COUNT(*) FROM (SELECT player_id, event_date FROM player_days GROUP BY 1,2 HAVING COUNT(*) > 1)") == 0
    assert one(conn, "SELECT COUNT(*) FROM spend_daily WHERE channel = 'organic'") == 0  # organic has no spend
    assert one(conn, "SELECT COUNT(*) FROM players WHERE channel = 'organic'") > 0


def test_foreign_keys_and_day_numbers(conn):
    assert one(conn, "SELECT COUNT(*) FROM player_days d LEFT JOIN players p ON p.player_id = d.player_id WHERE p.player_id IS NULL") == 0
    bad = one(
        conn,
        "SELECT COUNT(*) FROM player_days d JOIN players p ON p.player_id = d.player_id "
        "WHERE CAST(julianday(d.event_date) - julianday(p.install_date) AS INTEGER) != d.day_n",
    )
    assert bad == 0


def test_every_event_is_reported_after_it_happens(conn):
    assert one(conn, "SELECT COUNT(*) FROM player_days WHERE reported_at <= event_date") == 0
    assert one(conn, "SELECT COUNT(*) FROM players WHERE reported_at <= install_date") == 0
    assert one(conn, "SELECT COUNT(*) FROM spend_daily WHERE reported_at <= spend_date") == 0


def test_declared_completeness_is_never_earlier_than_any_row_it_covers(conn):
    """complete_at(n) must be >= reported_at of every player and every day row with day_n <= n of that cohort."""
    for n in (3, 30, 180):
        late_days = one(
            conn,
            """SELECT COUNT(*) FROM player_days d JOIN players p ON p.player_id = d.player_id
               JOIN cohort_reporting c ON c.channel = p.channel AND c.install_date = p.install_date AND c.through_day = ?
               WHERE d.day_n <= ? AND d.reported_at > c.complete_at""",
            n, n,
        )
        late_players = one(
            conn,
            """SELECT COUNT(*) FROM players p JOIN cohort_reporting c
               ON c.channel = p.channel AND c.install_date = p.install_date AND c.through_day = ?
               WHERE p.reported_at > c.complete_at""",
            n,
        )
        assert late_days == 0 and late_players == 0, f"through_day {n}"


def test_reporting_lag_ranges_follow_config(conn, dataset):
    cfg = dataset["cfg"]
    lags = pd.read_sql_query(
        "SELECT p.channel, MIN(julianday(d.reported_at) - julianday(d.event_date)) AS lo, "
        "MAX(julianday(d.reported_at) - julianday(d.event_date)) AS hi "
        "FROM player_days d JOIN players p ON p.player_id = d.player_id GROUP BY p.channel",
        conn,
    ).set_index("channel")
    for ch in PAID:
        lag = cfg["channels"][ch]["report_lag"]
        assert lags.loc[ch, "lo"] == lag["min_days"]
        assert lags.loc[ch, "hi"] == lag["min_days"] + lag["jitter_days"]


def test_milestones_only_exist_when_the_outcome_is_inside_storage(conn, dataset):
    horizon = dataset["cfg"]["horizon_days"]
    for n in (3, 30, 180):
        assert one(conn, "SELECT COUNT(*) FROM cohort_reporting WHERE through_day = ? AND channel = 'meta'", n) == horizon - n


def test_there_are_inactive_players_in_every_channel(conn):
    """Players with no row after day 0 exist, so denominators that ignore them would overstate retention."""
    df = pd.read_sql_query(
        "SELECT p.channel, COUNT(*) AS inactive FROM players p WHERE NOT EXISTS "
        "(SELECT 1 FROM player_days d WHERE d.player_id = p.player_id AND d.day_n >= 1) GROUP BY p.channel",
        conn,
    )
    assert len(df) == 6 and (df.inactive > 0).all()
