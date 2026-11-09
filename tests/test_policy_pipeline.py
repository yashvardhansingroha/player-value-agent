"""Policy on generated data (30%-volume dataset): decision-date-only information, historical invariance of decisions, payback maturity.
Selected tests; they do not prove every leakage path is closed."""
import json
import shutil
import sqlite3

import pandas as pd
import pytest

from src.data.calendar import day_to_date
from src.forecast.pipeline import START, run_backtest, run_review
from src.ledger.proposal import jsonable
from src.policy import policy as pol
from src.policy.constraints import check_constraints, load_forecast_config, load_policy
from tests.test_forecast_pipeline import small_cfg
from tests.test_historical_invariance import mutate_not_yet_available

DAY = 252
AS_OF = day_to_date(START, DAY)


def small_policy():
    p = load_policy()
    p["payback"]["min_mature_installs"] = 300  # 30% volume: cohorts are ~3x smaller
    return p


def small_forecast_cfg():
    return small_cfg(DAY)


@pytest.fixture(scope="module")
def env(small_dataset, tmp_path_factory):
    out = tmp_path_factory.mktemp("policy")
    run_backtest(small_dataset["db"], out / "f.db", small_forecast_cfg())
    return {"db": small_dataset["db"], "store": out / "f.db", "dir": out}


def decide(db, store, day=DAY):
    dec, st = sqlite3.connect(db), sqlite3.connect(store)
    try:
        as_of = day_to_date(START, day)
        budgets = pol.initial_budgets_from_spend(dec, as_of)
        ceiling = sum(budgets.values()) * 1.2
        res = pol.decide_all(dec, st, as_of, budgets, ceiling, small_policy(), small_forecast_cfg())
        return res, budgets, ceiling
    finally:
        dec.close()
        st.close()


def test_decisions_on_generated_data_record_every_stage_and_respect_the_hard_constraints(env):
    res, budgets, ceiling = decide(env["db"], env["store"])
    assert set(res["decisions"]) == {"meta", "google_uac", "applovin", "unity", "tiktok"}
    assert res["constraint_violations"] == [] and res["totals"]["after"] <= ceiling + 1e-9
    assert res["totals"]["reserve_before"] == pytest.approx(ceiling - sum(budgets.values()))
    for ch, d in res["decisions"].items():
        names = [s["stage"] for s in d["stages"]]
        assert names[:2] == ["evidence_binding", "completeness"] and "forecast_coverage" in names and "uncertainty" in names and "payback_target" in names
        assert d["evidence"]["forecast_evidence_binding"]["ok"] is True
        sched = d["evidence"]["forecast_evidence_binding"]["run_schedule"]  # scheduling is shown next to, not inside, the evidence hash
        assert sched["schedule"] == small_forecast_cfg()["backtest"] and "forecasting and quality settings only" in sched["note"]
        assert round(d["proposed_budget"], 6) == d["proposed_budget"]  # proposed budgets are exact to 6 decimals (no float noise)
        assert d["action"] in ("increase_test", "hold", "decrease", "insufficient_evidence") and d["reasons"]
        assert (d["proposed_budget"] != d["current_budget"]) == (d["action"] in ("increase_test", "decrease"))
        assert "ONLY the included cohorts" in d["evidence"]["populations"]["note"]
    p = pol.build_rules_proposal(res, 1, "h")
    assert all(i.qualification == "plan_required" for i in p.items if i.action == "increase_test")
    assert check_constraints(budgets, {i.channel: i.new_budget for i in p.items}, ceiling, small_policy()) == []


def test_forecast_evidence_from_another_configuration_is_refused_on_generated_data(env):
    """The stored summaries were written under the original forecast configuration. Run the review under summary_min_calibration = 1000."""
    changed = small_forecast_cfg()
    changed["forecast"]["summary_min_calibration"] = 1000
    dec, st = sqlite3.connect(env["db"]), sqlite3.connect(env["store"])
    try:
        budgets = pol.initial_budgets_from_spend(dec, AS_OF)
        ceiling = sum(budgets.values()) * 1.2
        res = pol.decide_all(dec, st, AS_OF, budgets, ceiling, small_policy(), changed)
        ok = pol.decide_all(dec, st, AS_OF, budgets, ceiling, small_policy(), small_forecast_cfg())
    finally:
        dec.close()
        st.close()
    for ch, d in res["decisions"].items():
        b = d["evidence"]["forecast_evidence_binding"]
        assert d["action"] == "insufficient_evidence" and b["code"] == "forecast_evidence_config_mismatch" and b["stored_hash"] != b["expected_hash"]
        assert d["proposed_budget"] == d["current_budget"]
    assert res["policy_hash"] != ok["policy_hash"]  # proposals carry the hash of the configuration that produced them
    assert pol.build_rules_proposal(res, 1).policy_hash == res["policy_hash"]
    assert all(d["evidence"]["forecast_evidence_binding"]["ok"] for d in ok["decisions"].values())  # matching configuration is accepted
    assert res["inverse_violations"] == [] and ok["inverse_violations"] == []


def test_payback_target_uses_only_cohorts_whose_d180_milestone_was_complete(env):
    res, _, _ = decide(env["db"], env["store"])
    seen = 0
    for d in res["decisions"].values():
        t = d["evidence"].get("payback_target")
        if t and t.get("target_d30_roas") is not None:
            seen += 1
            gap = (pd.Timestamp(AS_OF) - pd.Timestamp(t["last_install"])).days
            assert gap >= 180 + 1 + 1  # D180 elapsed plus at least the minimum 1-day reporting lag
            assert t["target_d30_roas"] == pytest.approx(1.0 / t["d180_over_d30"]) and t["sum_d180"] >= t["sum_d30"]
    assert seen >= 3, "expected most channels to have enough mature history at this date"


def test_decisions_are_invariant_to_not_yet_available_records_and_to_later_store_rows(env, tmp_path):
    base = jsonable(decide(env["db"], env["store"])[0])
    # (a) alter / delete / add decision-database records that are reported AFTER the as-of date
    mutated = tmp_path / "mut.db"
    shutil.copyfile(env["db"], mutated)
    mutate_not_yet_available(mutated, AS_OF)
    assert jsonable(decide(mutated, env["store"])[0]) == base
    # (b) a store that also holds LATER reviews (the forecast store is a backtest artifact spanning many dates)
    later = tmp_path / "later.db"
    shutil.copyfile(env["store"], later)
    dec, st = sqlite3.connect(env["db"]), sqlite3.connect(later)
    for d in (DAY + 7, DAY + 14):
        run_review(dec, st, day_to_date(START, d), small_forecast_cfg())
    dec.close()
    st.close()
    assert jsonable(decide(env["db"], later)[0]) == base


def test_control_changing_visible_history_changes_the_payback_evidence(env, tmp_path):
    base = decide(env["db"], env["store"])[0]
    edited = tmp_path / "vis.db"
    shutil.copyfile(env["db"], edited)
    c = sqlite3.connect(edited)  # an UNSUPPORTED edit of already-visible facts, only to show the invariance tests can fail
    c.execute(f"UPDATE player_days SET ad_revenue = ad_revenue * 3 WHERE day_n > 3 AND reported_at <= '{AS_OF}'")
    c.commit()
    c.close()
    changed = decide(edited, env["store"])[0]
    pt = lambda r: {ch: d["evidence"]["payback_target"].get("target_d30_roas") for ch, d in r["decisions"].items()}
    assert pt(base) != pt(changed)


def test_excluded_populations_use_decision_date_facts_only(env):
    res, _, _ = decide(env["db"], env["store"])
    allowed = {"install_date", "reason", "installs", "spend", "observed_d0_3_net_revenue"}
    n = 0
    for d in res["decisions"].values():
        for c in d["evidence"]["populations"]["excluded_withheld"]["cohorts"]:
            n += 1
            assert set(c) == allowed
    assert n > 0  # at least one channel had a withheld cohort in its window at this review
