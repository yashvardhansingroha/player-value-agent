"""Direct tests of the availability views. Current metrics are also gated by milestone completeness, which can mask a
broken row-level filter, so the views themselves are checked against independent counts from the base tables."""
import pytest

from src.data.availability import set_as_of
from src.data.calendar import day_to_date

START = "2025-01-06"


def count(conn, sql, *args):
    return conn.execute(sql, args).fetchone()[0]


@pytest.mark.parametrize("as_of_day", [100, 252, 308])
def test_views_contain_only_visible_rows_and_exactly_the_visible_rows(conn, as_of_day):
    t = day_to_date(START, as_of_day)
    set_as_of(conn, t)

    assert count(conn, "SELECT COUNT(*) FROM v_players WHERE reported_at > ?", t) == 0
    assert count(conn, "SELECT COUNT(*) FROM v_players") == count(conn, "SELECT COUNT(*) FROM main.players WHERE reported_at <= ?", t)

    assert count(conn, "SELECT COUNT(*) FROM v_player_days WHERE reported_at > ?", t) == 0
    expected_days = count(
        conn,
        "SELECT COUNT(*) FROM main.player_days d JOIN main.players p ON p.player_id = d.player_id "
        "WHERE d.reported_at <= ? AND p.reported_at <= ?", t, t,
    )
    assert count(conn, "SELECT COUNT(*) FROM v_player_days") == expected_days
    # no player-day may belong to a player who is not yet attributed
    assert count(conn, "SELECT COUNT(*) FROM v_player_days d LEFT JOIN v_players p ON p.player_id = d.player_id WHERE p.player_id IS NULL") == 0

    assert count(conn, "SELECT COUNT(*) FROM v_spend WHERE reported_at > ?", t) == 0
    assert count(conn, "SELECT COUNT(*) FROM v_spend") == count(conn, "SELECT COUNT(*) FROM main.spend_daily WHERE reported_at <= ?", t)

    assert count(conn, "SELECT COUNT(*) FROM v_cohort_reporting WHERE complete_at > ?", t) == 0
    assert count(conn, "SELECT COUNT(*) FROM v_cohort_reporting") == count(conn, "SELECT COUNT(*) FROM main.cohort_reporting WHERE complete_at <= ?", t)


def test_player_day_view_never_multiplies_rows(conn):
    set_as_of(conn, day_to_date(START, 336))
    assert count(conn, "SELECT COUNT(*) FROM (SELECT player_id, event_date FROM v_player_days GROUP BY 1, 2 HAVING COUNT(*) > 1)") == 0


def test_view_follows_a_changed_as_of(conn):
    set_as_of(conn, day_to_date(START, 100))
    early = count(conn, "SELECT COUNT(*) FROM v_player_days")
    set_as_of(conn, day_to_date(START, 300))
    late = count(conn, "SELECT COUNT(*) FROM v_player_days")
    set_as_of(conn, day_to_date(START, 100))
    assert late > early and count(conn, "SELECT COUNT(*) FROM v_player_days") == early


def test_nothing_is_visible_before_anything_is_reported(conn):
    set_as_of(conn, START)  # day 0 at 00:00: minimum lag is 1 day, so nothing can be visible yet
    for view in ("v_players", "v_player_days", "v_spend", "v_cohort_reporting"):
        assert count(conn, f"SELECT COUNT(*) FROM {view}") == 0, view
