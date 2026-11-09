"""Independent re-implementation of the cohort metrics in plain pandas/dicts, compared with the SQL on the full
generated dataset at several as-of dates. Two implementations must agree on every cohort and every column."""
import pandas as pd
import pytest

from src.data.calendar import day_to_date
from src.data.metrics import cohort_metrics, cohort_outcomes, store_fee

START = "2025-01-06"
AS_OF_DAYS = [245, 280, 308]


def load_raw(conn):
    return {
        "players": pd.read_sql_query("SELECT * FROM players", conn),
        "days": pd.read_sql_query("SELECT * FROM player_days", conn),
        "spend": pd.read_sql_query("SELECT * FROM spend_daily", conn),
        "rep": pd.read_sql_query("SELECT * FROM cohort_reporting", conn),
    }


def oracle_cohort_metrics(raw, as_of, fee):
    players = raw["players"][raw["players"].reported_at <= as_of]
    days = raw["days"][raw["days"].reported_at <= as_of].merge(
        players[["player_id", "channel", "install_date"]], on="player_id", how="inner", validate="many_to_one"
    )
    spend = raw["spend"][raw["spend"].reported_at <= as_of].set_index(["channel", "spend_date"]).spend.to_dict()
    ready = set(map(tuple, raw["rep"][(raw["rep"].complete_at <= as_of) & (raw["rep"].through_day == 3)][["channel", "install_date"]].to_numpy()))
    installs = players.groupby(["channel", "install_date"]).size().to_dict()
    d3 = days[days.day_n <= 3]
    g = d3.groupby(["channel", "install_date"])
    sums = g[["iap_gross", "ad_revenue", "sessions", "levels"]].sum()
    n_rows = g.size()
    payers = d3[d3.iap_gross > 0].groupby(["channel", "install_date"]).player_id.nunique()
    d1 = days[days.day_n == 1].groupby(["channel", "install_date"]).size()
    d3_only = days[days.day_n == 3].groupby(["channel", "install_date"]).size()

    out = []
    for key in sorted(set(installs) | set(spend)):
        n_inst, sp, done = installs.get(key, 0), spend.get(key), key in ready
        rec = dict(channel=key[0], install_date=key[1], installs_reported=n_inst, spend=sp, d3_complete=int(done))
        nan = float("nan")
        gross = sums.iap_gross.get(key, 0.0)
        ad = sums.ad_revenue.get(key, 0.0)
        rows = n_rows.get(key, 0)
        full = done and n_inst > 0
        rec["cpi"] = sp / n_inst if full and sp is not None else nan
        rec["d1_retention"] = d1.get(key, 0) / n_inst if full else nan
        rec["d3_retention"] = d3_only.get(key, 0) / n_inst if full else nan
        rec["d0_3_net_revenue"] = gross * (1 - fee) + ad if done else nan
        rec["d0_3_gross_iap"] = gross if done else nan
        rec["d0_3_ad_revenue"] = ad if done else nan
        rec["payer_rate_d0_3"] = payers.get(key, 0) / n_inst if full else nan
        rec["arpdau_d0_3"] = (gross * (1 - fee) + ad) / rows if done and rows > 0 else nan
        rec["sessions_per_active_user"] = sums.sessions.get(key, 0) / rows if done and rows > 0 else nan
        rec["levels_per_active_user"] = sums.levels.get(key, 0) / rows if done and rows > 0 else nan
        out.append(rec)
    return pd.DataFrame(out)


def oracle_outcomes(raw, as_of, n, fee):
    players = raw["players"][raw["players"].reported_at <= as_of]
    days = raw["days"][raw["days"].reported_at <= as_of].merge(
        players[["player_id", "channel", "install_date"]], on="player_id", how="inner", validate="many_to_one"
    )
    spend = raw["spend"][raw["spend"].reported_at <= as_of].set_index(["channel", "spend_date"]).spend.to_dict()
    ready = raw["rep"][(raw["rep"].complete_at <= as_of) & (raw["rep"].through_day == n)]
    installs = players.groupby(["channel", "install_date"]).size().to_dict()
    sums = days[days.day_n <= n].groupby(["channel", "install_date"])[["iap_gross", "ad_revenue"]].sum()
    out = []
    for ch, d in sorted(zip(ready.channel, ready.install_date)):
        gross = sums.iap_gross.get((ch, d), 0.0)
        ad = sums.ad_revenue.get((ch, d), 0.0)
        out.append(dict(channel=ch, install_date=d, installs=installs.get((ch, d), 0), spend=spend.get((ch, d)),
                        gross_iap=gross, ad_revenue=ad, net_revenue=gross * (1 - fee) + ad))
    return pd.DataFrame(out)


@pytest.fixture(scope="module")
def raw(dataset):
    import sqlite3

    c = sqlite3.connect(dataset["db"])
    data = load_raw(c)
    fee = store_fee(c)
    c.close()
    return data, fee


@pytest.mark.parametrize("as_of_day", AS_OF_DAYS)
def test_cohort_metrics_match_independent_pandas_oracle(conn, raw, as_of_day):
    data, fee = raw
    as_of = day_to_date(START, as_of_day)
    sql = cohort_metrics(conn, as_of)
    oracle = oracle_cohort_metrics(data, as_of, fee)
    assert len(sql) == len(oracle) > 0
    pd.testing.assert_frame_equal(sql, oracle[sql.columns], check_dtype=False, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("through_day", [3, 30, 180])
def test_cohort_outcomes_match_independent_pandas_oracle(conn, raw, through_day):
    data, fee = raw
    as_of = day_to_date(START, 308)
    sql = cohort_outcomes(conn, as_of, through_day)
    oracle = oracle_outcomes(data, as_of, through_day, fee)
    assert len(sql) == len(oracle) > 0
    pd.testing.assert_frame_equal(sql, oracle[sql.columns], check_dtype=False, rtol=1e-9, atol=1e-9)


def test_spend_and_installs_are_conserved_by_the_cohort_join(conn, raw):
    """A join must never multiply acquisition spend or installs: totals equal the visible base-table totals."""
    data, _ = raw
    as_of = day_to_date(START, 280)
    df = cohort_metrics(conn, as_of)
    visible_spend = data["spend"][data["spend"].reported_at <= as_of].spend.sum()
    visible_players = (data["players"].reported_at <= as_of).sum()
    assert df.spend.sum() == pytest.approx(visible_spend)
    assert df.installs_reported.sum() == visible_players
    assert not df.duplicated(["channel", "install_date"]).any()
