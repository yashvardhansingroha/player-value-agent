"""SIMULATED experiment store, clock and hidden-side simulator: determinism, availability, freeze-before-outcomes, truth separation."""
import re
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from src.experiments import design as D
from src.experiments import lifecycle as L
from src.experiments import plan as P
from src.experiments import store as E
from src.experiments.scoring import replicate_study
from src.experiments.simulator import ExperimentSimulator, load_sim_config
from tests.experiment_helpers import CFG, manual_plan, synthetic_truth

ROOT = Path(__file__).resolve().parents[1]
AS_OF = "2025-09-08"
SEED = 12345


def make_sim(units, gamma=0.0, scale=1.0, unit_seed=3, truth=None):
    return ExperimentSimulator(truth or synthetic_truth(gamma=gamma), "meta", {u["unit_id"]: u["baseline_daily_budget"] for u in units}, 1.2, load_sim_config(),
                               scale=scale, unit_seed=unit_seed)


@pytest.fixture()
def env(tmp_path):
    units = D.make_units("meta", 90.0, 24, 1)
    sim = make_sim(units)
    ec = E.create_store(tmp_path / "e.db")
    E.set_clock(ec, AS_OF)
    E.add_units(ec, "meta", units)
    E.insert_observations(ec, sim.pre_rows(D.windows(AS_OF, 2, 14, 14), SEED))
    plan = manual_plan(units=units, as_of=AS_OF, treatment_days=14)
    return {"ec": ec, "sim": sim, "units": units, "plan": plan, "path": tmp_path / "e.db"}


def test_daily_observations_do_not_depend_on_how_the_clock_was_advanced(env):
    sim, plan = env["sim"], env["plan"]
    h = P.plan_hash(plan)
    all_at_once = sim.treatment_rows_up_to(plan, h, "2026-06-01", SEED, set())
    step1 = sim.treatment_rows_up_to(plan, h, "2025-10-12", SEED, set())
    done = {r["acquisition_date"] for r in step1}
    step2 = sim.treatment_rows_up_to(plan, h, "2026-06-01", SEED, done)
    key = lambda r: (r["unit_id"], r["acquisition_date"])
    assert sorted(step1 + step2, key=key) == sorted(all_at_once, key=key)  # per-day seeding: the same numbers either way
    assert len(all_at_once) == 24 * 14 and 0 < len(step1) < len(all_at_once)
    assert sim.pre_rows(D.windows(AS_OF, 2, 14, 14), SEED) == sim.pre_rows(D.windows(AS_OF, 2, 14, 14), SEED)
    assert sim.treatment_rows_up_to(plan, h, "2026-06-01", SEED + 1, set()) != all_at_once  # a different root seed is different noise


def test_outcomes_are_released_only_when_available_and_pending_ones_do_not_exist(env):
    ec, sim, plan = env["ec"], env["sim"], env["plan"]
    h = E.register_plan(ec, plan, CFG)
    w = plan["windows"]
    assert E.treatment_rows(ec, h) == 0
    first = L.advance_clock(ec, sim, plan, P.add_days(w["treatment_start"], 32), SEED)  # day 0's D30 + lag (2) has just elapsed
    assert first["new_rows"] == 24 and first["total_treatment_rows"] == 24  # exactly the first acquisition day, for every unit
    mid = L.advance_clock(ec, sim, plan, P.add_days(w["treatment_start"], 36), SEED)
    assert mid["total_treatment_rows"] == 24 * 5
    assert L.advance_clock(ec, sim, plan, P.add_days(w["treatment_start"], 36), SEED)["new_rows"] == 0  # idempotent
    done = L.advance_clock(ec, sim, plan, w["readout_available_on"], SEED)
    assert done["total_treatment_rows"] == 24 * 14


def test_the_database_itself_refuses_pending_outcomes_unfrozen_plans_and_a_backwards_clock(env):
    ec, sim, plan = env["ec"], env["sim"], env["plan"]
    h = P.plan_hash(plan)
    rows = sim.treatment_rows_up_to(plan, h, "2026-06-01", SEED, set())
    row, other = rows[0], rows[-1]  # two different unit-days, so the last check cannot be a primary-key clash
    with pytest.raises(sqlite3.IntegrityError, match="frozen first"):
        E.insert_observations(ec, [row])  # the plan has not been frozen yet: a treatment outcome cannot exist
    E.register_plan(ec, plan, CFG)
    with pytest.raises(sqlite3.IntegrityError, match="not available yet"):
        E.insert_observations(ec, [row])  # frozen now, but this day's D30 outcome is still pending at clock 2025-09-08
    E.set_clock(ec, "2025-12-31")
    E.insert_observations(ec, [row])  # frozen and available: accepted
    with pytest.raises(sqlite3.IntegrityError, match="backwards"):
        E.set_clock(ec, "2025-09-09")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        ec.execute("UPDATE observations SET spend = 0")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        ec.execute("DELETE FROM experiment_plans")
    with pytest.raises(sqlite3.IntegrityError, match="frozen first"):
        E.insert_observations(ec, [dict(other, plan_hash="0" * 64)])  # observation tied to a plan that was never frozen


def test_register_plan_refuses_invalid_plans_and_stamps_the_freeze_date(env):
    ec, plan = env["ec"], env["plan"]
    bad = dict(plan, units=plan["units"][:-1])
    with pytest.raises(ValueError, match="structurally valid"):
        E.register_plan(ec, bad, CFG)
    weak = manual_plan(units=env["units"], as_of=AS_OF, treatment_days=14, adequate=False)  # real power 30% ...
    weak["power_assessment"]["standards"]["target_power"] = 0.01  # ... passed off as adequate by lowering the standard
    weak["power_assessment"]["chosen_treatment_days"] = 14
    weak["power_assessment"]["adequate"], weak["power_assessment"]["reasons"] = P.adequacy_from_numbers(weak["power_assessment"])
    with pytest.raises(ValueError, match="does not follow the experiment configuration"):
        E.register_plan(ec, weak, CFG)  # the store checks against the authoritative configuration too
    h = E.register_plan(ec, plan, CFG)
    assert E.get_plan(ec, h) == plan and ec.execute("SELECT frozen_on FROM experiment_plans").fetchone()[0] == AS_OF
    with pytest.raises(sqlite3.IntegrityError):
        E.register_plan(ec, plan, CFG)  # the same plan cannot be frozen twice


def test_the_experiment_database_holds_only_observable_columns(env):
    cols = {t: {r[1] for r in env["ec"].execute(f"PRAGMA table_info({t})")} for t in ("units", "observations", "experiment_plans")}
    assert cols["units"] == {"unit_id", "channel", "baseline_daily_budget"}
    assert cols["observations"] == {"unit_id", "acquisition_date", "phase", "spend", "installs", "d1_active", "d30_net_revenue", "available_on", "generated_on", "plan_hash"}
    blob = env["path"].read_bytes()
    for needle in (b"rho", b"quality_gamma", b"true_iroas", b"hidden"):
        assert needle not in blob


def test_decision_facing_experiment_modules_never_import_the_simulator_scoring_or_hidden_truth():
    forbidden = re.compile(r"experiments\.simulator|experiments import .*simulator|experiments\.scoring|experiments import .*scoring|hidden_truth|load_hidden|experiment_sim")
    for name in ("plan.py", "design.py", "readout.py", "store.py", "lifecycle.py", "__init__.py"):
        text = (ROOT / "src" / "experiments" / name).read_text(encoding="utf-8")
        code = "\n".join(l for l in text.splitlines() if not l.strip().startswith(("#", '"""')))
        assert not forbidden.search(code), name
    for d in ("ledger", "policy", "forecast", "quality", "data", "agent", "validate"):
        for p in (ROOT / "src" / d).rglob("*.py"):
            assert "experiments.simulator" not in p.read_text(encoding="utf-8") and "experiments.scoring" not in p.read_text(encoding="utf-8"), p


def test_truth_functions_behave_sensibly(env):
    units = env["units"]
    flat = make_sim(units, gamma=0.0).true_effect(env["plan"])
    diluted = make_sim(units, gamma=0.8).true_effect(env["plan"])
    assert flat["true_iroas_treated_units"] > 0 and diluted["true_iroas_treated_units"] < flat["true_iroas_treated_units"]  # quality dilution lowers the return
    assert flat["true_extra_spend_per_day"] == pytest.approx(9.0 * 0.97, rel=1e-9)  # the 9/day step, 97% delivered
    big = make_sim(D.make_units("meta", 90.0 * 10, 24, 1), scale=10.0)  # 10x channel: budgets and installs scaled together
    assert big.expected_installs(big.b).sum() == pytest.approx(make_sim(units).expected_installs(make_sim(units).b).sum() * 10, rel=0.05)
    regime = synthetic_truth()
    regime["planted"]["regime_shift"]["start_day"] = 0  # every day after the shift: value x (0.7 x 0.7 + 1.1 x 0.3)
    shifted = make_sim(units, truth=regime)
    assert shifted.regime_factor("2025-09-08") == pytest.approx(0.7 * 0.7 + 1.1 * 0.3)


def test_replicate_study_reports_keep_rate_bias_and_coverage(env):
    plan = manual_plan(units=env["units"], as_of=AS_OF, treatment_days=14, target=0.05)
    r = replicate_study(plan, make_sim(env["units"]), n=25, seed=1)
    assert r["replicates"] == 25 and 0.0 <= r["keep_rate"] <= 1.0 and r["keep_rate_mc_se"] >= 0
    assert set(r) >= {"estimate_mean", "estimate_bias", "estimate_sd", "lower_bound_coverage_of_truth", "nominal_one_sided_coverage", "rollback_reasons", "true_iroas_treated_units"}
    assert np.isfinite(r["estimate_mean"]) and "SIMULATED" in r["label"]
