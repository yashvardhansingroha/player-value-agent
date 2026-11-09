"""Do the window checks fire on the dev-seed data where an outage is known to exist, and stay quiet elsewhere?
This is EVALUATION-side QA: it uses the known planted dates (decision-facing code does not). The false-alarm sweep covers every
review date and channel, so an over-sensitive detector would fail here too. Dev seed only; not held-out evidence."""
import pytest

from src.data.calendar import day_to_date
from src.data.metrics import daily_activity
from src.forecast.pipeline import START, load_config
from src.quality.checks import check_tracking_gap

Q = load_config()["quality"]
MONDAYS = list(range(245, 337, 7))
LAG = {"meta": 2, "google_uac": 2, "applovin": 2, "unity": 9, "tiktok": 2}  # declared lag per channel (from the config)


def gap(conn, channel, day):
    as_of = day_to_date(START, day)
    return check_tracking_gap(daily_activity(conn, as_of, channel), LAG[channel], as_of, Q)


@pytest.mark.parametrize("day", [301, 308])
def test_planted_outage_is_flagged_while_it_is_visible(conn, day):
    res = gap(conn, "google_uac", day)
    assert res.evaluable and not res.passed, res.reason  # outage on event days 296..302


def test_no_false_alarm_on_any_other_review_date_or_channel(conn):
    false_alarms = []
    for channel in LAG:
        for day in MONDAYS:
            if channel == "google_uac" and day in (301, 308):
                continue
            res = gap(conn, channel, day)
            if res.evaluable and not res.passed:
                false_alarms.append((channel, day, res.reason))
    assert not false_alarms, false_alarms
