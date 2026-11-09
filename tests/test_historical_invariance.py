"""Historical-invariance tests. Metrics are RECOMPUTED from the database for an earlier as_of on (a) the original DB
and (b) a copy whose NOT-YET-AVAILABLE records (reported_at / complete_at after that as_of) were altered, deleted, or
added. Results must be identical. Positive controls prove the mutations are substantive.

SCOPE (see docs/data_contract.md): v1 supports IMMUTABLE facts that arrive late. These tests do NOT show that history is
safe under corrections. Overwriting or deleting a fact that was ALREADY visible at an earlier as_of cannot be
reconstructed from this schema (it keeps no record versions); the positive-control tests below demonstrate that such edits
change recomputed history. Supporting corrections would require retained record versions or historical snapshots."""
import shutil
import sqlite3

import pandas as pd
import pytest

from src.data.availability import set_as_of
from src.data.calendar import day_to_date
from src.data.metrics import cohort_metrics, cohort_outcomes, retention_curve

START = "2025-01-06"
CHANNELS = ["meta", "google_uac", "applovin", "unity", "tiktok", "organic"]


def day(n):
    return day_to_date(START, n)


def snapshot(conn, as_of):
    """Everything decision logic will read at `as_of`, all recomputed from scratch."""
    parts = {"metrics": cohort_metrics(conn, as_of)}
    for n in (3, 30, 180):
        parts[f"outcomes_{n}"] = cohort_outcomes(conn, as_of, n)
    for ch in CHANNELS:
        parts[f"retention_{ch}"] = retention_curve(conn, as_of, ch)
    return parts


def assert_same(a, b):
    assert a.keys() == b.keys()
    for k in a:
        pd.testing.assert_frame_equal(a[k], b[k], check_exact=True, obj=k)


def mutate_not_yet_available(path, as_of):
    """Alter every record whose availability date is AFTER as_of."""
    c = sqlite3.connect(path)
    c.executescript(
        f"""
        UPDATE player_days SET iap_gross = iap_gross * 100, ad_revenue = ad_revenue * 50 WHERE reported_at > '{as_of}';
        DELETE FROM player_days WHERE reported_at > '{as_of}' AND player_id % 2 = 0;
        UPDATE spend_daily SET spend = spend * 10 WHERE reported_at > '{as_of}';
        UPDATE cohort_reporting SET complete_at = '2099-01-01' WHERE complete_at > '{as_of}';
        -- late-attributed duplicates of OLD installs: install_date in the past, but reported in the future
        INSERT INTO players SELECT player_id + 1000000000, channel, install_date, '2099-01-01' FROM players WHERE player_id % 7 = 0;
        INSERT INTO player_days SELECT player_id + 1000000000, event_date, day_n, sessions, levels, ad_impressions,
               iap_gross * 10, ad_revenue, '2099-01-01' FROM player_days WHERE player_id % 7 = 0;
        """
    )
    c.commit()
    c.close()


@pytest.mark.parametrize("as_of_day", [252, 280, 308])
def test_changing_not_yet_available_records_does_not_change_an_earlier_view(dataset, tmp_path, as_of_day):
    as_of = day(as_of_day)
    copy = tmp_path / "mutated.db"
    shutil.copyfile(dataset["db"], copy)
    mutate_not_yet_available(copy, as_of)

    original, mutated = sqlite3.connect(dataset["db"]), sqlite3.connect(copy)
    try:
        assert_same(snapshot(original, as_of), snapshot(mutated, as_of))

        # positive control: the same mutation DOES change what is seen once time moves past the altered records
        later = day(as_of_day + 60)
        before, after = snapshot(original, later), snapshot(mutated, later)
        assert not before["metrics"].equals(after["metrics"]), "mutation was not substantive"
        assert not before["outcomes_30"].equals(after["outcomes_30"])
    finally:
        original.close()
        mutated.close()


def test_changing_day4_plus_rows_does_not_change_d3_metrics_but_changes_d30_outcomes(dataset, tmp_path):
    as_of = day(280)
    copy = tmp_path / "d4.db"
    shutil.copyfile(dataset["db"], copy)
    c = sqlite3.connect(copy)
    c.execute(f"UPDATE player_days SET iap_gross = iap_gross * 100 WHERE day_n > 3 AND reported_at <= '{as_of}'")
    c.commit()
    c.close()

    original, mutated = sqlite3.connect(dataset["db"]), sqlite3.connect(copy)
    try:
        pd.testing.assert_frame_equal(cohort_metrics(original, as_of), cohort_metrics(mutated, as_of), check_exact=True)
        pd.testing.assert_frame_equal(cohort_outcomes(original, as_of, 3), cohort_outcomes(mutated, as_of, 3), check_exact=True)
        # control: D30 outcomes legitimately use day 4+ rows, so they must change
        assert not cohort_outcomes(original, as_of, 30).equals(cohort_outcomes(mutated, as_of, 30))
    finally:
        original.close()
        mutated.close()


def test_changing_already_visible_rows_does_change_the_view(dataset, tmp_path):
    """Control for the whole file: tampering with VISIBLE history is detected."""
    as_of = day(280)
    copy = tmp_path / "visible.db"
    shutil.copyfile(dataset["db"], copy)
    c = sqlite3.connect(copy)
    c.execute(f"UPDATE player_days SET iap_gross = iap_gross * 2 WHERE day_n <= 3 AND reported_at <= '{as_of}'")
    c.commit()
    c.close()
    original, mutated = sqlite3.connect(dataset["db"]), sqlite3.connect(copy)
    try:
        assert not cohort_metrics(original, as_of).equals(cohort_metrics(mutated, as_of))
    finally:
        original.close()
        mutated.close()


def test_results_are_recomputed_not_cached_across_as_of_changes(conn):
    first = cohort_metrics(conn, day(252))
    other = cohort_metrics(conn, day(308))
    again = cohort_metrics(conn, day(252))
    assert not first.equals(other)
    pd.testing.assert_frame_equal(first, again, check_exact=True)


def test_late_arriving_previously_absent_event_is_invisible_before_arrival_and_counted_after(dataset, tmp_path):
    """INSERTION of a previously absent, immutable event that is reported late (no overwrite, no delete).

    Target: a meta player installed on day 270 who has no day-2 row. The event (a 500.0 purchase on event day 272) is
    reported on day 274, which is within the cohort's declared D3 completeness date (day 275), so the data contract holds.
    """
    install_day, event_day, arrival_day = 270, 272, 274
    early, later = day(arrival_day - 1), day(276)  # 273: before arrival; 276: after arrival and after D3 completeness (275)
    copy = tmp_path / "late.db"
    shutil.copyfile(dataset["db"], copy)

    original, mutated = sqlite3.connect(dataset["db"]), sqlite3.connect(copy)
    try:
        pid = original.execute(
            "SELECT p.player_id FROM players p WHERE p.channel = 'meta' AND p.install_date = ? "
            "AND NOT EXISTS (SELECT 1 FROM player_days d WHERE d.player_id = p.player_id AND d.day_n = 2) "
            "ORDER BY p.player_id LIMIT 1", (day(install_day),),
        ).fetchone()[0]
        absent = "SELECT COUNT(*) FROM player_days WHERE player_id = ? AND event_date = ?"
        # (1) the target player-day does not exist before insertion, in either database
        assert original.execute(absent, (pid, day(event_day))).fetchone()[0] == 0
        assert mutated.execute(absent, (pid, day(event_day))).fetchone()[0] == 0
        before_rows = mutated.execute("SELECT COUNT(*) FROM player_days").fetchone()[0]

        mutated.execute("INSERT INTO player_days VALUES (?, ?, 2, 3, 3, 3, 500.0, 0.0, ?)", (pid, day(event_day), day(arrival_day)))
        mutated.commit()
        assert mutated.execute("SELECT COUNT(*) FROM player_days").fetchone()[0] == before_rows + 1  # pure insertion
        assert mutated.execute(absent, (pid, day(event_day))).fetchone()[0] == 1

        # (2) an earlier view, recomputed from scratch, is unchanged
        assert_same(snapshot(original, early), snapshot(mutated, early))
        seen = "SELECT COUNT(*) FROM v_player_days WHERE player_id = ? AND event_date = ?"
        set_as_of(mutated, early)
        assert mutated.execute(seen, (pid, day(event_day))).fetchone()[0] == 0  # not visible before arrival
        set_as_of(mutated, day(arrival_day))
        assert mutated.execute(seen, (pid, day(event_day))).fetchone()[0] == 1  # visible from 00:00 of the arrival date

        # (3) a later view includes the event: exactly one cohort changes, by exactly the late purchase
        a, b = cohort_metrics(original, later), cohort_metrics(mutated, later)
        assert list(a.columns) == list(b.columns) and len(a) == len(b)
        changed = ~((a == b) | (a.isna() & b.isna())).all(axis=1)
        assert changed.sum() == 1
        ra, rb = a[changed].iloc[0], b[changed].iloc[0]
        assert (rb.channel, rb.install_date) == ("meta", day(install_day))
        assert rb.d3_complete == 1
        assert rb.d0_3_gross_iap == pytest.approx(ra.d0_3_gross_iap + 500.0)
        assert rb.d0_3_net_revenue == pytest.approx(ra.d0_3_net_revenue + 500.0 * 0.7)
    finally:
        original.close()
        mutated.close()
