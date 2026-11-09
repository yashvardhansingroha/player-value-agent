"""Metric SQL checked against a tiny hand-built database whose answers are computed by hand below.
Includes inactive players (they must stay in every retention denominator)."""
import pandas as pd
import pytest

from src.data.load import create_database
from src.data.metrics import cohort_metrics, cohort_outcomes, retention_curve

import sqlite3

FEE = 0.30
D0 = "2025-01-06"  # install date of the main cohort
ROW = dict(sessions=2, levels=3, ad_impressions=0)


def day(n):
    return f"2025-01-{6 + n:02d}"


def make_rows():
    # (player, day_n, iap_gross, ad_revenue, reported_at)
    spec = [
        (1, 0, 10.0, 0.5, "2025-01-07"), (1, 1, 0.0, 0.2, "2025-01-08"), (1, 3, 0.0, 0.1, "2025-01-10"),
        (1, 4, 100.0, 1.0, "2025-01-11"),                        # D4 row: must NOT enter D3 metrics
        (2, 0, 0.0, 0.3, "2025-01-07"), (2, 1, 20.0, 0.2, "2025-01-08"),
        (3, 0, 0.0, 0.1, "2025-01-07"), (4, 0, 0.0, 0.1, "2025-01-07"),
        (5, 0, 0.0, 0.1, "2025-01-07"), (6, 0, 0.0, 0.1, "2025-01-07"),
        (3, 2, 0.0, 9.0, "2025-01-09"),                          # reported exactly on 2025-01-09 (boundary test)
    ]
    return pd.DataFrame(
        [dict(player_id=p, event_date=day(n), day_n=n, iap_gross=i, ad_revenue=a, reported_at=r, **ROW) for p, n, i, a, r in spec]
    )


@pytest.fixture()
def mini(tmp_path):
    # 10 meta players installed on D0: players 7..10 have NO rows at all (inactive), 4..6 only a D0 row.
    players = pd.DataFrame(
        [dict(player_id=i, channel="meta", install_date=D0, reported_at="2025-01-07") for i in range(1, 11)]
        # a second cohort on the same day in another channel, and a third meta cohort with spend but no players yet
        + [dict(player_id=11, channel="unity", install_date=D0, reported_at="2025-01-07")]
    )
    days = make_rows()
    days = pd.concat([days, pd.DataFrame([dict(player_id=11, event_date=D0, day_n=0, iap_gross=5.0, ad_revenue=0.0,
                                               reported_at="2025-01-07", **ROW)])], ignore_index=True)
    spend = pd.DataFrame(
        [
            dict(channel="meta", spend_date=D0, spend=40.0, reported_at="2025-01-07"),
            dict(channel="unity", spend_date=D0, spend=7.0, reported_at="2025-01-07"),
            dict(channel="meta", spend_date="2025-01-07", spend=11.0, reported_at="2025-01-08"),
        ]
    )
    reporting = pd.DataFrame(
        [
            dict(channel="meta", install_date=D0, through_day=3, complete_at="2025-01-11"),
            dict(channel="meta", install_date=D0, through_day=30, complete_at="2025-02-10"),
            dict(channel="unity", install_date=D0, through_day=3, complete_at="2025-01-11"),
        ]
    )
    path = tmp_path / "mini.db"
    create_database(path, players, days, spend, reporting, {"store_fee": str(FEE), "start_date": D0})
    c = sqlite3.connect(path)
    yield c
    c.close()


def row(df, channel, date):
    r = df[(df.channel == channel) & (df.install_date == date)]
    assert len(r) == 1, "exactly one row per cohort"
    return r.iloc[0]


def test_d3_metrics_by_hand(mini):
    r = row(cohort_metrics(mini, "2025-01-11"), "meta", D0)
    assert r.installs_reported == 10 and r.d3_complete == 1
    assert r.spend == 40.0 and r.cpi == pytest.approx(4.0)
    # D1 active: players 1, 2 of ALL 10 installed (inactive players 3..10 stay in the denominator)
    assert r.d1_retention == pytest.approx(2 / 10)
    assert r.d3_retention == pytest.approx(1 / 10)
    # D0..D3 gross IAP = 10 + 20 = 30 (the D4 purchase of 100 is excluded); ad = .5+.2+.1+.3+.2+.1*4 + 9.0 (player 3, day 2)
    ad = 0.5 + 0.2 + 0.1 + 0.3 + 0.2 + 0.1 * 4 + 9.0
    assert r.d0_3_gross_iap == pytest.approx(30.0) and r.d0_3_ad_revenue == pytest.approx(ad)
    assert r.d0_3_net_revenue == pytest.approx(30.0 * (1 - FEE) + ad)
    assert r.payer_rate_d0_3 == pytest.approx(2 / 10)
    rows_d3 = 10  # active player-days with day_n <= 3
    assert r.arpdau_d0_3 == pytest.approx((30.0 * (1 - FEE) + ad) / rows_d3)
    assert r.sessions_per_active_user == pytest.approx(2.0) and r.levels_per_active_user == pytest.approx(3.0)


def test_one_row_per_cohort_and_no_join_multiplication(mini):
    df = cohort_metrics(mini, "2025-01-11")
    assert not df.duplicated(["channel", "install_date"]).any()
    assert len(df) == 3  # meta D0, unity D0, meta 2025-01-07 (spend visible, no players)
    assert df.spend.sum() == pytest.approx(40.0 + 7.0 + 11.0)  # each spend row counted exactly once
    assert df.installs_reported.sum() == 11
    # cohort with spend but no installed players yet: present, zero installs, D3 metrics NULL
    r = row(df, "meta", "2025-01-07")
    assert r.installs_reported == 0 and r.spend == 11.0 and r.d3_complete == 0 and pd.isna(r.d1_retention)


def test_incomplete_d3_gives_null_metrics_not_partial_numbers(mini):
    r = row(cohort_metrics(mini, "2025-01-10"), "meta", D0)  # complete_at is 2025-01-11
    assert r.d3_complete == 0 and r.installs_reported == 10 and r.spend == 40.0
    for col in ("cpi", "d1_retention", "d3_retention", "d0_3_net_revenue", "payer_rate_d0_3", "arpdau_d0_3"):
        assert pd.isna(r[col]), col


def test_as_of_boundary_is_inclusive_and_future_rows_are_hidden(mini):
    # Force the D3 milestone to look complete so the revenue columns are populated at early dates; this isolates
    # the row-level reported_at filter from the completeness filter (which has its own test above).
    mini.execute("UPDATE cohort_reporting SET complete_at = '2025-01-01' WHERE through_day = 3")

    def ad(as_of):
        return row(cohort_metrics(mini, as_of), "meta", D0).d0_3_ad_revenue

    # visible on 01-08: p1 D0 .5, p1 D1 .2, p2 D0 .3, p2 D1 .2, and four D0 rows of .1  -> 1.6
    assert ad("2025-01-08") == pytest.approx(1.6)
    # player 3's day-2 row (ad 9.0) has reported_at == 2025-01-09: visible exactly ON that date (<=), not the day before
    assert ad("2025-01-09") == pytest.approx(1.6 + 9.0)
    # player 1's D3 row (ad .1) is reported 2025-01-10; the D4 row (reported 01-11) never counts toward D3 metrics
    assert ad("2025-01-10") == pytest.approx(1.6 + 9.0 + 0.1)
    assert ad("2025-02-01") == pytest.approx(1.6 + 9.0 + 0.1)


def test_outcomes_include_day4_and_respect_milestone_completeness(mini):
    assert cohort_outcomes(mini, "2025-01-20", 30).empty  # D30 milestone not complete yet
    out = cohort_outcomes(mini, "2025-02-10", 30)
    r = row(out, "meta", D0)
    gross = 10 + 20 + 100
    ad = 0.5 + 0.2 + 0.1 + 1.0 + 0.3 + 0.2 + 0.1 * 4 + 9.0
    assert r.installs == 10 and r.spend == 40.0
    assert r.net_revenue == pytest.approx(gross * (1 - FEE) + ad)
    with pytest.raises(ValueError):
        cohort_outcomes(mini, "2025-02-10", 7)  # only declared milestones are allowed


def test_retention_curve_denominator_includes_all_installed_players(mini):
    curve = retention_curve(mini, "2025-02-10", "meta").set_index("day_n")
    assert list(curve.index) == list(range(31))  # all of D0..D30, including days with no activity rows
    assert (curve.installs == 10).all()  # 10 installed, even though 4 never appear in player_days
    assert curve.loc[0, "retention"] == pytest.approx(6 / 10)  # only 6 players have a D0 row in this hand-built data
    assert curve.loc[1, "retention"] == pytest.approx(2 / 10)
    assert curve.loc[4, "active_players"] == 1
    assert curve.loc[10, "active_players"] == 0 and curve.loc[10, "retention"] == 0.0  # empty day is zero, not missing


def build_db(tmp_path, players, days, reporting, as_of_unused=None):
    """Tiny database for retention edge cases. players: list of ids (all channel 'meta', installed D0)."""
    p = pd.DataFrame([dict(player_id=i, channel="meta", install_date=D0, reported_at="2025-01-07") for i in players])
    d = pd.DataFrame(
        [dict(player_id=pid, event_date=day(n), day_n=n, iap_gross=0.0, ad_revenue=0.0, reported_at=day(n + 1), **ROW) for pid, n in days],
        columns=["player_id", "event_date", "day_n", "sessions", "levels", "ad_impressions", "iap_gross", "ad_revenue", "reported_at"],
    )
    s = pd.DataFrame([dict(channel="meta", spend_date=D0, spend=1.0, reported_at="2025-01-07")])
    r = pd.DataFrame(reporting, columns=["channel", "install_date", "through_day", "complete_at"])
    path = tmp_path / "edge.db"
    create_database(path, p, d, s, r, {"store_fee": str(FEE), "start_date": D0})
    return sqlite3.connect(path)


D30_COMPLETE = [dict(channel="meta", install_date=D0, through_day=30, complete_at="2025-02-10")]


def test_retention_mature_cohort_with_no_activity_returns_31_zero_days(tmp_path):
    c = build_db(tmp_path, players=[1, 2, 3, 4, 5], days=[], reporting=D30_COMPLETE)  # 5 installed, ZERO activity rows
    curve = retention_curve(c, "2025-02-10", "meta")
    assert list(curve.day_n) == list(range(31))
    assert (curve.active_players == 0).all() and (curve.installs == 5).all()
    assert (curve.retention == 0.0).all() and (curve.retention_available == 1).all()  # zero because players exist
    c.close()


def test_retention_mature_cohort_with_gaps_between_active_days(tmp_path):
    days = [(1, 0), (1, 2), (1, 5), (2, 0), (2, 5), (3, 0)]  # player 4 never active; nobody active on days 1, 3, 4, 6..30
    c = build_db(tmp_path, players=[1, 2, 3, 4], days=days, reporting=D30_COMPLETE)
    curve = retention_curve(c, "2025-02-10", "meta").set_index("day_n")
    assert len(curve) == 31 and (curve.installs == 4).all()
    expected = {0: 3 / 4, 2: 1 / 4, 5: 2 / 4}
    for n in range(31):
        assert curve.loc[n, "retention"] == pytest.approx(expected.get(n, 0.0)), f"day {n}"
    c.close()


def test_retention_is_unavailable_not_zero_when_no_eligible_cohort(tmp_path):
    # (a) players exist but the D30 milestone is not complete yet
    c = build_db(tmp_path, players=[1, 2, 3], days=[(1, 0)], reporting=D30_COMPLETE)
    early = retention_curve(c, "2025-01-20", "meta")
    assert list(early.day_n) == list(range(31)) and (early.installs == 0).all()
    assert early.retention.isna().all() and (early.retention_available == 0).all()
    assert (early.active_players == 0).all()
    # (b) a channel with no cohorts at all
    other = retention_curve(c, "2025-02-10", "unity")
    assert other.retention.isna().all() and (other.retention_available == 0).all()
    c.close()


def test_retention_is_unavailable_when_milestone_complete_but_no_player_visible(tmp_path):
    # cohort milestone row is visible but every install attribution is reported later than as_of
    c = build_db(tmp_path, players=[1, 2], days=[], reporting=D30_COMPLETE)
    c.execute("UPDATE players SET reported_at = '2025-03-01'")
    c.commit()
    curve = retention_curve(c, "2025-02-10", "meta")
    assert (curve.installs == 0).all() and curve.retention.isna().all() and (curve.retention_available == 0).all()
    c.close()
