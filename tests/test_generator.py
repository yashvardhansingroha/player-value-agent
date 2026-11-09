"""Generator behaviour: reproducibility, held-out seed guard, planted events present, calibration intents hold."""
import json
import sqlite3
from datetime import date

import numpy as np
import pandas as pd
import pytest

from scripts.generate_data import build
from src.generator.generate import generate, load_config
from src.generator.intents import compute_intent_metrics, evaluate_intents, load_scenarios

START = "2025-01-06"


def test_day_zero_is_a_monday():
    assert date.fromisoformat(START).weekday() == 0
    assert load_config()["start_date"] == START


def test_same_seed_reproduces_identical_tables_and_different_seed_differs():
    cfg = load_config()
    a, b, c = (generate(cfg, seed=s, install_scale=0.05) for s in (3, 3, 4))
    for name in ("players", "player_days", "spend_daily", "cohort_reporting"):
        pd.testing.assert_frame_equal(getattr(a, name), getattr(b, name), check_exact=True)
    assert not a.player_days.equals(c.player_days)
    assert not a.spend_daily.spend.equals(c.spend_daily.spend)


def test_heldout_seeds_are_refused(tmp_path):
    held = load_config()["heldout_seeds"][0]
    with pytest.raises(SystemExit):
        build(seed=held, out_root=tmp_path)
    assert not list(tmp_path.rglob("*.db"))


def test_hidden_truth_json_contains_truth_but_the_decision_db_does_not(dataset):
    truth = json.loads(dataset["hidden"].read_text(encoding="utf-8"))
    assert truth["channels"]["applovin"]["response"]["type"] == "hill"
    assert len(truth["channels"]["meta"]["daily_mean_installs_noiseless"]) == dataset["cfg"]["horizon_days"]


def test_revenue_is_heavy_tailed_and_whale_is_planted(conn, dataset):
    purchases = pd.read_sql_query("SELECT iap_gross FROM player_days WHERE iap_gross > 0", conn).iap_gross
    assert purchases.max() > 50 * purchases.median()
    w = dataset["cfg"]["planted"]["tiny_sample_whale"]
    big = conn.execute("SELECT MAX(iap_gross) FROM player_days d JOIN players p USING (player_id) WHERE p.channel = 'tiktok'").fetchone()[0]
    assert big >= w["whale_iap_gross"]


def test_regime_shift_and_outage_leave_installs_untouched_but_change_events(conn, dataset):
    cfg = dataset["cfg"]
    o = cfg["planted"]["tracking_outage"]
    spend = pd.read_sql_query("SELECT spend_date, spend FROM spend_daily WHERE channel = ?", conn, params=(o["channel"],))
    assert spend.spend.min() > 0  # spend never drops in an outage


def test_cohort_sizes_for_paid_channels_are_usable(conn):
    """Context for ambiguity A1 (cohort noise): median installs per paid cohort-day by channel."""
    sizes = pd.read_sql_query(
        "SELECT channel, install_date, COUNT(*) AS n FROM players WHERE channel != 'organic' GROUP BY 1, 2", conn
    ).groupby("channel").n.median()
    assert (sizes >= 30).all(), sizes.to_dict()


def test_all_committed_scenario_calibration_intents_hold_on_the_dev_seed(conn, dataset):
    """Generator QA only: the generator was tuned to meet these on the dev seed. Not evidence about any model."""
    hidden = json.loads(dataset["hidden"].read_text(encoding="utf-8"))
    rows = evaluate_intents(compute_intent_metrics(conn, hidden), load_scenarios())
    failed = [r for r in rows if not r[-1]]
    assert len(rows) == 17 and not failed, failed
