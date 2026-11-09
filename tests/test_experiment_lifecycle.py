"""SIMULATED lifecycle through the real ledger: plan -> approval -> application -> pending -> readout -> keep or rollback.

These tests exercise LIFECYCLE MECHANICS with a manual plan whose power numbers are fabricated (so they are fast). They say nothing about
statistical performance, which is measured by the real planning study and the evaluation-side replicate study in the Phase 4 report.
The experiment-backed increases here are NOT mechanics_test increases: they go through the real plan gate.

`execute_readout` is AUTHORITATIVE: it recomputes the readout itself and ignores any caller-supplied decision (a supplied readout is only
a preview that must match). The regressions below fabricate keeps and rollbacks, truncate the data, edit decisions and mislink changes."""
import copy

import numpy as np
import pytest

from src.experiments import design as D
from src.experiments import lifecycle as L
from src.experiments import plan as P
from src.experiments import readout as R
from src.experiments import store as E
from src.experiments.simulator import ExperimentSimulator, load_sim_config
from src.ledger.ledger import Ledger
from src.ledger.proposal import Proposal, ProposalItem
from src.policy.constraints import load_forecast_config, load_policy
from tests.experiment_helpers import manual_plan, synthetic_truth

POLICY, FORECAST, EXPERIMENT = load_policy(), load_forecast_config(), D.load_config()
AS_OF, SEED, SCALE = "2025-09-08", 12345, 50.0  # scale 50: large enough volumes that the targets below decide the outcome unambiguously
BUDGETS = {"meta": 90.0 * SCALE, "google_uac": 80.0 * SCALE, "applovin": 100.0 * SCALE, "unity": 60.0 * SCALE, "tiktok": 70.0 * SCALE}


def build_env(tmp_path, target, treatment_days=14, scale=SCALE):
    led = Ledger(tmp_path / "ledger.db", POLICY, FORECAST)
    led.initialize(BUDGETS, now=AS_OF)
    baseline, proposed = BUDGETS["meta"], round(BUDGETS["meta"] * 1.1, 6)
    units = D.make_units("meta", baseline, 24, 1)
    sim = ExperimentSimulator(synthetic_truth(), "meta", {u["unit_id"]: u["baseline_daily_budget"] for u in units}, 1.2, load_sim_config(), scale=scale, unit_seed=3)
    ec = E.create_store(tmp_path / "exp.db")
    E.set_clock(ec, AS_OF)
    E.add_units(ec, "meta", units)
    E.insert_observations(ec, sim.pre_rows(D.windows(AS_OF, 2, 14, treatment_days), SEED))
    plan = manual_plan(baseline=baseline, proposed=proposed, units=units, policy_hash=led.policy_hash, as_of=AS_OF, treatment_days=treatment_days, target=target,
                       experiment_config_hash=D.config_hash(EXPERIMENT))
    return {"led": led, "ec": ec, "sim": sim, "plan": plan, "baseline": baseline, "proposed": proposed}


def policy_style_proposal(led, env):
    items = [ProposalItem(channel=c, action="increase_test" if c == "meta" else "hold", current_budget=b, new_budget=env["proposed"] if c == "meta" else b,
                          qualification="plan_required" if c == "meta" else "none", reason="policy output") for c, b in led.state()["budgets"].items()]
    return Proposal(proposal_id="P-policy", as_of=AS_OF, ledger_version=led.head(), policy_hash=led.policy_hash, items=items)


def freeze_and_apply(env, start=True):
    led, ec, plan = env["led"], env["ec"], env["plan"]
    h = led.register_frozen_plan(plan, "designer", AS_OF)
    assert E.register_plan(ec, plan, EXPERIMENT) == h  # the same canonical content gets the same hash in both places
    prop = L.experiment_proposal(policy_style_proposal(led, env), "meta", plan, led.head())
    led.create_proposal(prop, AS_OF)
    assert led.approve(prop.proposal_id, "reviewer", AS_OF)["ok"]
    applied = led.apply(prop.proposal_id, AS_OF)
    assert applied["ok"] and applied["kind"] == "APPLY"  # a real experiment-backed increase, NOT MECHANICS_TEST_APPLY
    if start:
        L.start_experiment(led, ec, h, applied["change_id"])
    return h, prop, applied


def release_all(env):
    return L.advance_clock(env["ec"], env["sim"], env["plan"], env["plan"]["windows"]["readout_available_on"], SEED)


def do_readout(env):
    """A PREVIEW: the production readout function on the store's own clock and observations."""
    obs = E.load_observations(env["ec"])
    return R.run_readout(env["plan"], obs, E.clock(env["ec"]), POLICY, FORECAST, EXPERIMENT)


def execute(env, preview=None, who="readout rule", **cfgs):
    kw = {"policy_cfg": POLICY, "forecast_cfg": FORECAST, "experiment_cfg": EXPERIMENT, **cfgs}
    return L.execute_readout(env["led"], env["ec"], P.plan_hash(env["plan"]), who, preview=preview, **kw)


def outcomes(env):
    return env["ec"].execute("SELECT COUNT(*) FROM experiment_outcomes").fetchone()[0]


def assert_refused_and_nothing_changed(res, code, env, before):
    assert res["status"] == "rejected" and res["code"] == code and res["outcome_recorded"] is False, res
    assert env["led"].state() == before and outcomes(env) == 0  # no budget moved, no completed outcome recorded
    events = [e["event"] for e in env["led"].audit()]
    assert "rollback_succeeded" not in events and "experiment_kept" not in events


def fabricated(env, decision):
    """What a dishonest caller might hand in: the right plan hash and the decision it wants executed."""
    return {"label": "SIMULATED READOUT. ALL DATA IS SYNTHETIC.", "plan_hash": P.plan_hash(env["plan"]), "clock": E.clock(env["ec"]), "decision": decision,
            "decision_code": "keep" if decision == "keep" else "rollback:lower_bound_below_target",
            "estimate": {"iroas": 9.99, "se": 0.01, "lower_bound": 9.9, "alpha": 0.1, "df": 22}, "rule": {"target_d30_roas": 0.05, "lower_bound_meets_target": decision == "keep"}}


def test_an_inadequate_design_turns_the_action_into_insufficient_evidence_with_the_budget_unchanged():
    decision = {"channel": "meta", "action": "increase_test", "current_budget": 90.0, "proposed_budget": 99.0, "reasons": ["lower bound clears target"],
                "increase_candidate": {"s1": 99.0}}
    good = L.decision_after_design(decision, manual_plan(adequate=True))
    assert good["action"] == "increase_test" and good["proposed_budget"] == 99.0 and good["experiment_design"]["adequate"] is True
    bad_plan = manual_plan(adequate=False)
    bad = L.decision_after_design(decision, bad_plan)
    assert bad["action"] == "insufficient_evidence" and bad["proposed_budget"] == 90.0 and bad["increase_candidate"] is None
    assert bad["reasons"][0].startswith("experiment design is inadequate") and bad["experiment_design"]["adequate"] is False
    assert decision["action"] == "increase_test"  # the input is not mutated


def test_experiment_proposal_changes_only_the_experiment_channel():
    led_items = [ProposalItem(channel=c, action="decrease" if c == "tiktok" else "increase_test" if c == "meta" else "hold", current_budget=100.0,
                              new_budget=90.0 if c == "tiktok" else 110.0 if c == "meta" else 100.0, qualification="plan_required" if c == "meta" else "none", reason="r")
                 for c in ("meta", "tiktok", "unity")]
    src = Proposal(proposal_id="P-src", as_of=AS_OF, ledger_version=1, policy_hash="p" * 64, items=led_items)
    plan = manual_plan(baseline=100.0, proposed=110.0, policy_hash="p" * 64)
    exp = L.experiment_proposal(src, "meta", plan, 1)
    by = {i.channel: i for i in exp.items}
    assert by["meta"].qualification == "experiment_backed" and by["meta"].experiment_plan_hash == P.plan_hash(plan)
    assert by["tiktok"].action == "hold" and by["tiktok"].new_budget == 100.0 and by["unity"].action == "hold"  # the unrelated decrease is NOT bundled
    with pytest.raises(ValueError):
        L.experiment_proposal(src, "tiktok", plan, 1)  # not an increase test


def test_full_lifecycle_rollback_path(tmp_path):
    env = build_env(tmp_path, target=5.0)  # an unreachable target: the readout must say rollback
    led, ec, plan = env["led"], env["ec"], env["plan"]
    before = led.state()
    h, prop, applied = freeze_and_apply(env)
    assert led.state()["budgets"]["meta"] == env["proposed"] and led.state()["total"] == pytest.approx(before["total"] + (env["proposed"] - env["baseline"]))

    # PENDING: nothing has matured; an early readout fails explicitly and no outcome rows exist
    status = R.readout_status(plan, E.clock(ec), E.load_observations(ec))
    assert status["state"] == "pending" and status["days_remaining"] > 40 and len(status["pending_acquisition_days"]) == 14
    with pytest.raises(R.ReadoutNotAvailable, match="not available"):
        do_readout(env)
    # part-way: some days have matured, the experiment is still pending and still refuses
    part = L.advance_clock(ec, env["sim"], plan, P.add_days(plan["windows"]["treatment_start"], 38), SEED)
    assert 0 < part["total_treatment_rows"] < 24 * 14
    with pytest.raises(R.ReadoutNotAvailable) as exc:
        do_readout(env)
    assert exc.value.days_remaining > 0 and len(exc.value.pending_days) < 14

    # READOUT at the available date
    release_all(env)
    assert E.clock(ec) == plan["windows"]["readout_available_on"]
    out = do_readout(env)
    assert out["decision"] == "rollback" and out["decision_code"] == "rollback:lower_bound_below_target" and "SIMULATED" in out["label"]
    assert out["spending"]["planned_extra_spend_per_day_total"] == pytest.approx(env["proposed"] - env["baseline"])  # planned: exactly the approved step
    assert 0.85 < out["spending"]["delivery_fraction"] < 1.05 and out["spending"]["realised_extra_spend_per_day_total_estimate"] != out["spending"]["planned_extra_spend_per_day_total"]
    assert out["input_problems"] == {} and out["invalid_readout"] is False

    res = execute(env, preview=out)
    assert res["status"] == "rolled_back" and led.state()["budgets"] == before["budgets"] and led.state()["reserve"] == pytest.approx(before["reserve"])
    assert res["readout"]["decision"] == "rollback" and ec.execute("SELECT outcome FROM experiment_outcomes WHERE plan_hash = ?", (h,)).fetchone()[0] == "rolled_back"
    events = [e["event"] for e in led.audit()]
    assert events.index("plan_registered") < events.index("approval_recorded") < events.index("apply_succeeded") < events.index("rollback_succeeded")
    # DUPLICATE ACTION: a second execution is refused and nothing changes again
    again = execute(env, preview=out)
    assert again["code"] == "readout_already_executed" and led.head() == 3 and outcomes(env) == 1


def test_full_lifecycle_keep_path(tmp_path):
    env = build_env(tmp_path, target=0.05)  # an easy target: the readout should keep
    led, ec, plan = env["led"], env["ec"], env["plan"]
    h, _, applied = freeze_and_apply(env)
    release_all(env)
    out = do_readout(env)
    assert out["decision"] == "keep" and out["decision_code"] == "keep" and out["rule"]["lower_bound_meets_target"] is True
    res = execute(env)  # no preview needed: the decision is recomputed
    assert res["status"] == "kept" and led.state()["budgets"]["meta"] == env["proposed"] and led.head() == 2  # the increase stays; no rollback version
    assert led.audit("experiment_kept")[-1]["change_id"] == applied["change_id"]
    assert execute(env)["code"] == "readout_already_executed"


# ---------------------------------------------------------------- authoritative execution (independent-review regressions)


def test_fabricated_early_keep_or_rollback_is_refused_and_changes_nothing(tmp_path):
    """REGRESSION: a caller-supplied result with the right plan hash and decision='keep' used to be executed even though the clock was
    before the readout date and no treatment observation existed. Neither a fabricated keep nor a fabricated rollback may do anything."""
    env = build_env(tmp_path, target=0.05)
    ec, plan = env["ec"], env["plan"]
    freeze_and_apply(env)
    assert E.treatment_rows(ec, P.plan_hash(plan)) == 0  # no outcomes exist at all
    before = env["led"].state()
    for stage in ("at application", "part-way"):
        if stage == "part-way":
            L.advance_clock(ec, env["sim"], plan, P.add_days(plan["windows"]["treatment_start"], 38), SEED)
        for decision in ("keep", "rollback"):
            for preview in (fabricated(env, decision), None):
                res = execute(env, preview=preview)
                assert_refused_and_nothing_changed(res, "readout_not_available", env, before)
                assert res["days_remaining"] > 0 and res["available_on"] == plan["windows"]["readout_available_on"]
    assert env["led"].state()["budgets"]["meta"] == env["proposed"]  # the increase is still in place, untouched by any refusal
    assert [r[0] for r in ec.execute("SELECT event FROM experiment_events WHERE event = 'readout_execution_rejected'")]  # the refusals are on the record


def test_incomplete_observations_cannot_be_read_out_or_executed(tmp_path):
    env = build_env(tmp_path, target=5.0)
    led, ec, plan = env["led"], env["ec"], env["plan"]
    freeze_and_apply(env)
    release_all(env)
    before = led.state()
    ec.execute("DROP TRIGGER observations_no_delete")  # simulate a damaged store (normally blocked)
    ec.execute("DELETE FROM observations WHERE unit_id = ? AND acquisition_date = ?", (plan["units"][0]["unit_id"], plan["windows"]["treatment_start"]))
    for preview in (None, fabricated(env, "rollback"), fabricated(env, "keep")):
        assert_refused_and_nothing_changed(execute(env, preview=preview), "readout_incomplete", env, before)


def test_a_modified_decision_after_a_valid_readout_is_rejected_and_only_the_recomputed_decision_is_ever_executed(tmp_path):
    env = build_env(tmp_path, target=5.0)  # the honest answer is rollback
    freeze_and_apply(env)
    release_all(env)
    before = env["led"].state()
    valid = do_readout(env)
    assert valid["decision"] == "rollback"
    flipped = dict(valid, decision="keep", decision_code="keep")
    res = execute(env, preview=flipped)
    assert_refused_and_nothing_changed(res, "readout_preview_mismatch", env, before)
    assert "decision" in res["differing_fields"] and "decision_code" in res["differing_fields"]
    padded = copy.deepcopy(valid)
    padded["estimate"]["lower_bound"] += 100.0  # a doctored number with the right decision still does not match
    assert_refused_and_nothing_changed(execute(env, preview=padded), "readout_preview_mismatch", env, before)
    assert_refused_and_nothing_changed(execute(env, preview=dict(valid, plan_hash="0" * 64)), "readout_preview_mismatch", env, before)
    # a genuine preview, or none at all, executes the RECOMPUTED decision
    done = execute(env, preview=valid)
    assert done["status"] == "rolled_back" and done["readout"]["decision"] == "rollback" and env["led"].state()["budgets"] == {**before["budgets"], "meta": env["baseline"]}

    (tmp_path / "keep").mkdir()
    keep_env = build_env(tmp_path / "keep", target=0.05)  # and the other direction: a caller who wants a rollback of a justified keep
    freeze_and_apply(keep_env)
    release_all(keep_env)
    keep_before = keep_env["led"].state()
    keep_valid = do_readout(keep_env)
    assert keep_valid["decision"] == "keep"
    assert_refused_and_nothing_changed(execute(keep_env, preview=dict(keep_valid, decision="rollback", decision_code="rollback:lower_bound_below_target")),
                                       "readout_preview_mismatch", keep_env, keep_before)
    assert execute(keep_env)["status"] == "kept"


def mislink(env):
    """Apply the experiment's increase honestly, then an UNRELATED change on another channel. Returns (plan hash, real change, unrelated change)."""
    led = env["led"]
    h, _, applied = freeze_and_apply(env, start=False)
    st = led.state()
    items = [ProposalItem(channel=c, action="decrease" if c == "unity" else "hold", current_budget=b, new_budget=round(b * 0.95, 6) if c == "unity" else b, reason="unrelated")
             for c, b in st["budgets"].items()]
    led.create_proposal(Proposal(proposal_id="P-unrelated", as_of=AS_OF, ledger_version=st["version"], policy_hash=led.policy_hash, items=items), AS_OF)
    assert led.approve("P-unrelated", "reviewer", AS_OF)["ok"]
    return h, applied, led.apply("P-unrelated", AS_OF)


def test_an_incorrect_ledger_change_link_is_refused_at_start_and_at_execution(tmp_path):
    """REGRESSION: the run link was never checked against the ledger, so a readout could 'roll back' an unrelated change."""
    env = build_env(tmp_path, target=5.0)
    led, ec = env["led"], env["ec"]
    h, real, unrelated = mislink(env)
    assert unrelated["ok"]
    for wrong in (unrelated["change_id"], "C-9999-does-not-exist", "INIT"):
        with pytest.raises(ValueError, match="not the applied change"):
            L.start_experiment(led, ec, h, wrong)
    with pytest.raises(ValueError, match="cannot be started"):
        L.start_experiment(led, ec, "0" * 64, real["change_id"])  # a plan the ledger never registered
    assert ec.execute("SELECT COUNT(*) FROM experiment_runs").fetchone()[0] == 0
    # bypass start_experiment: write the wrong link straight into the store (what a buggy or hostile caller could do)
    ec.execute("INSERT INTO experiment_runs VALUES (?,?,?)", (h, unrelated["change_id"], AS_OF))
    release_all(env)
    before = led.state()
    res = execute(env)
    assert_refused_and_nothing_changed(res, "experiment_link_invalid", env, before)
    assert res["linked_change_id"] == unrelated["change_id"] and before["budgets"]["unity"] == round(BUDGETS["unity"] * 0.95, 6)  # the unrelated cut survives
    assert any("does not carry this plan's hash" in e for e in res["errors"])


def test_a_change_that_is_already_rolled_back_or_not_an_experiment_apply_cannot_be_linked(tmp_path):
    env = build_env(tmp_path, target=0.05)
    led, ec = env["led"], env["ec"]
    h, _, applied = freeze_and_apply(env, start=False)
    assert led.rollback(applied["change_id"], "operator", "manual rollback", AS_OF)["ok"]
    with pytest.raises(ValueError, match="rolled_back, not active"):
        L.start_experiment(led, ec, h, applied["change_id"])
    mech = Ledger(tmp_path / "mech.db", POLICY, FORECAST, allow_mechanics_test=True)
    mech.initialize(BUDGETS, now=AS_OF)
    st = mech.state()
    items = [ProposalItem(channel=c, action="increase_test" if c == "meta" else "hold", current_budget=b, new_budget=env["proposed"] if c == "meta" else b,
                          qualification="mechanics_test" if c == "meta" else "none", reason="mechanics") for c, b in st["budgets"].items()]
    mech.create_proposal(Proposal(proposal_id="P-mech", as_of=AS_OF, ledger_version=st["version"], policy_hash=mech.policy_hash, items=items), AS_OF)
    assert mech.approve("P-mech", "r", AS_OF)["ok"]
    m = mech.apply("P-mech", AS_OF)
    assert m["kind"] == "MECHANICS_TEST_APPLY"
    mech.register_frozen_plan(env["plan"], "designer", AS_OF)
    errors = mech.verify_experiment_change(m["change_id"], h)
    assert any("MECHANICS_TEST_APPLY" in e for e in errors) and any("does not carry this plan's hash" in e for e in errors)  # never an experiment-backed APPLY


def test_the_experiment_store_must_hold_exactly_the_ledgers_plan(tmp_path):
    env = build_env(tmp_path, target=5.0)
    led, ec, plan = env["led"], env["ec"], env["plan"]
    h = led.register_frozen_plan(plan, "designer", AS_OF)  # frozen in the ledger only; never in the experiment store
    prop = L.experiment_proposal(policy_style_proposal(led, env), "meta", plan, led.head())
    led.create_proposal(prop, AS_OF)
    assert led.approve(prop.proposal_id, "reviewer", AS_OF)["ok"]
    applied = led.apply(prop.proposal_id, AS_OF)
    with pytest.raises(ValueError, match="not frozen in the experiment store"):
        L.start_experiment(led, ec, h, applied["change_id"])
    before = led.state()
    assert_refused_and_nothing_changed(execute(env), "plan_not_frozen_in_experiment_store", env, before)


def test_rollback_blocked_when_a_later_change_touched_the_channel_and_it_is_not_treated_as_kept(tmp_path):
    env = build_env(tmp_path, target=5.0)
    led, ec, plan = env["led"], env["ec"], env["plan"]
    h, _, applied = freeze_and_apply(env)
    st = led.state()  # a later, unrelated approved change on the same channel (a decrease needs no plan)
    items = [ProposalItem(channel=c, action="decrease" if c == "meta" else "hold", current_budget=b, new_budget=round(b * 0.95, 6) if c == "meta" else b, reason="later") for c, b in st["budgets"].items()]
    later = Proposal(proposal_id="P-later", as_of="2025-10-01", ledger_version=st["version"], policy_hash=led.policy_hash, items=items)
    led.create_proposal(later, "2025-10-01")
    assert led.approve("P-later", "reviewer")["ok"] and led.apply("P-later", "2025-10-01")["ok"]
    release_all(env)
    out = do_readout(env)
    assert out["decision"] == "rollback"
    res = execute(env, preview=out)
    assert res["status"] == "rollback_blocked" and res["outcome_recorded"] is False and res["ledger"]["code"] == "channel_changed_since"
    assert "NOT treated as kept" in res["note"]
    assert outcomes(env) == 0  # no outcome recorded: it can be retried after resolution
    assert led.state()["budgets"]["meta"] == round(env["proposed"] * 0.95, 6)
    assert led.audit("experiment_rollback_not_executed")  # the failure to roll back is on the record


def test_configuration_must_be_explicitly_loaded_for_the_readout_and_the_rollback(tmp_path):
    env = build_env(tmp_path, target=5.0)
    led, ec, plan = env["led"], env["ec"], env["plan"]
    h, _, _ = freeze_and_apply(env)
    release_all(env)
    other_policy = copy.deepcopy(POLICY)
    other_policy["marginal"]["incremental_quality_haircut"] = 0.70
    with pytest.raises(R.ConfigurationMismatch, match="load the plan's configuration explicitly"):
        R.run_readout(plan, E.load_observations(ec), E.clock(ec), other_policy, FORECAST, EXPERIMENT)  # a different policy configuration is loaded
    other_exp = copy.deepcopy(EXPERIMENT)
    other_exp["estimator"]["alpha"] = 0.05
    with pytest.raises(R.ConfigurationMismatch):
        R.run_readout(plan, E.load_observations(ec), E.clock(ec), POLICY, FORECAST, other_exp)
    before = led.state()
    # the configurations SUPPLIED to the execution must be the plan's too
    assert_refused_and_nothing_changed(execute(env, experiment_cfg=other_exp), "configuration_mismatch", env, before)
    assert_refused_and_nothing_changed(execute(env, policy_cfg=other_policy), "configuration_mismatch", env, before)
    # the ledger's experiment configuration must be the plan's
    led.reload_config(POLICY, FORECAST, "2025-12-01", experiment_cfg=other_exp)
    assert_refused_and_nothing_changed(execute(env), "experiment_config_not_loaded", env, led.state())
    led.reload_config(POLICY, FORECAST, "2025-12-02", experiment_cfg=EXPERIMENT)
    # the ledger's policy configuration must be the plan's
    led.reload_config(other_policy, FORECAST, "2025-12-03")  # the ledger now holds a different configuration than the plan's
    res = execute(env)
    assert res["status"] == "configuration_not_loaded" and "reload_config" in res["action"] and outcomes(env) == 0 and res["outcome_recorded"] is False
    assert led.audit("experiment_readout_rejected")[-1]["details"]["code"] == "configuration_not_loaded"
    led.reload_config(POLICY, FORECAST, "2025-12-04")  # explicitly load the configuration the plan was frozen under
    assert execute(env)["status"] == "rolled_back"


def test_database_path_and_in_memory_replicate_path_give_identical_estimates(tmp_path):
    """The readout reads rows stored in the experiment database; the evaluation-side replicate study uses in-memory arrays from the same
    simulator. For the same seed the two must produce exactly the same estimate, or the replicate study would not describe the readout."""
    env = build_env(tmp_path, target=5.0)
    freeze_and_apply(env)
    release_all(env)
    out = do_readout(env)
    a = env["sim"].arrays(env["plan"], SEED)  # same root seed as the stored rows
    st = D.unit_stats(a["spend"], a["installs"], a["d1_active"], a["d30_net_revenue"], env["plan"]["windows"]["pre_days"])
    est = D.estimate(st, D.treat_mask(env["plan"], env["sim"].unit_ids), D.planned_mean_increment(env["plan"]), 0.10)
    assert est["iroas"][0] == pytest.approx(out["estimate"]["iroas"], rel=1e-12) and est["lower"][0] == pytest.approx(out["estimate"]["lower_bound"], rel=1e-12)
    assert est["se"][0] == pytest.approx(out["estimate"]["se"], rel=1e-12)
    other = env["sim"].arrays(env["plan"], SEED + 1)
    st2 = D.unit_stats(other["spend"], other["installs"], other["d1_active"], other["d30_net_revenue"], env["plan"]["windows"]["pre_days"])
    assert D.estimate(st2, D.treat_mask(env["plan"], env["sim"].unit_ids), D.planned_mean_increment(env["plan"]), 0.10)["iroas"][0] != pytest.approx(out["estimate"]["iroas"])


def test_readout_refuses_incomplete_observations_and_rows_that_belong_to_another_plan(tmp_path):
    env = build_env(tmp_path, target=5.0)
    led, ec, plan = env["led"], env["ec"], env["plan"]
    freeze_and_apply(env)
    release_all(env)
    obs = E.load_observations(ec)
    drop = obs[~((obs.unit_id == plan["units"][0]["unit_id"]) & (obs.phase == "treatment") & (obs.acquisition_date == plan["windows"]["treatment_start"]))]
    with pytest.raises(R.ReadoutIncomplete, match="unit-day rows"):
        R.run_readout(plan, drop, E.clock(ec), POLICY, FORECAST, EXPERIMENT)
    foreign = obs.copy()
    foreign.loc[foreign.phase == "treatment", "plan_hash"] = "f" * 64  # outcome rows generated for some OTHER plan are not this plan's outcomes
    with pytest.raises(R.ReadoutIncomplete, match="unit-day rows"):
        R.run_readout(plan, foreign, E.clock(ec), POLICY, FORECAST, EXPERIMENT)
    assert do_readout(env)["decision"] == "rollback"  # control: the real rows read out fine


def test_readout_uses_the_one_estimator_and_the_thresholds_frozen_in_the_plan(tmp_path, monkeypatch):
    env = build_env(tmp_path, target=0.05)
    freeze_and_apply(env)
    release_all(env)
    calls = []
    real_e, real_d = D.estimate, D.decide
    monkeypatch.setattr(D, "estimate", lambda *a, **k: (calls.append("estimate"), real_e(*a, **k))[1])
    monkeypatch.setattr(D, "decide", lambda *a, **k: (calls.append("decide"), real_d(*a, **k))[1])
    out = do_readout(env)
    assert calls == ["estimate", "decide"] and out["decision"] == "keep"  # exactly the planning functions, once, over one replicate
    calls.clear()
    assert execute(env)["status"] == "kept" and calls == ["estimate", "decide"]  # the execution recomputes through the same production path
    # the threshold comes from the frozen plan, not from a config: change the config's target-free settings and the plan's rule still governs
    loose = copy.deepcopy(env["plan"])
    loose["decision_rule"]["target_d30_roas"] = loose["target"]["target_d30_roas"] = 5.0
    assert D.rule_from_plan(loose)["target_d30_roas"] == 5.0 and D.rule_from_plan(env["plan"])["target_d30_roas"] == 0.05


# ---------------------------------------------------------------- invalid inputs fail closed (independent-review regression)


def break_cells(obs, column, value, how_many=1):
    broken = obs.copy()
    broken[column] = broken[column].astype(float)
    idx = broken.index[broken.phase == "treatment"][:how_many]
    broken.loc[idx, column] = value
    return broken


def test_missing_or_non_finite_observations_make_the_readout_invalid_and_never_keep(tmp_path):
    env = build_env(tmp_path, target=0.05)  # honest data KEEPS at this target
    freeze_and_apply(env)
    release_all(env)
    ec, plan = env["ec"], env["plan"]
    obs = E.load_observations(ec)
    run = lambda frame: R.run_readout(plan, frame, E.clock(ec), POLICY, FORECAST, EXPERIMENT)
    control = run(obs)
    assert control["decision"] == "keep" and control["input_problems"] == {} and control["invalid_readout"] is False
    for column in ("d1_active", "installs", "spend", "d30_net_revenue"):
        for bad in (np.nan, np.inf, -np.inf):
            out = run(break_cells(obs, column, bad))
            assert out["decision"] == "rollback" and out["decision_code"] == "rollback:invalid_readout_statistic" and out["invalid_readout"] is True, (column, bad)
            assert out["input_problems"].get(column, 0) >= 1
    # the review's scenario: the lower bound CLEARS the target, yet one unavailable guardrail input still forces a rollback (the old code kept)
    guardrail_only = run(break_cells(obs, "d1_active", np.nan))
    assert guardrail_only["rule"]["lower_bound_meets_target"] is True and guardrail_only["estimate"]["lower_bound"] >= guardrail_only["rule"]["target_d30_roas"]
    assert guardrail_only["decision"] == "rollback" and guardrail_only["guardrails"]["d1_retention_change"] is None
    # each guardrail on its own: an unavailable D1 retention (or CPI) cannot be read as a passing guardrail
    assert run(break_cells(obs, "d1_active", np.nan))["guardrails"]["d1_retention_change"] is None
    assert run(break_cells(obs, "installs", np.nan))["guardrails"]["cpi_ratio"] is None
    # impossible values are invalid too
    assert run(break_cells(obs, "spend", -5.0))["decision_code"] == "rollback:invalid_readout_statistic"
    assert run(break_cells(obs, "d30_net_revenue", -1.0))["decision_code"] == "rollback:invalid_readout_statistic"
    too_many_active = obs.copy()
    too_many_active.loc[too_many_active.index[too_many_active.phase == "treatment"][0], "d1_active"] = too_many_active["installs"].max() + 1000
    assert run(too_many_active)["decision_code"] == "rollback:invalid_readout_statistic"  # more D1-active players than installs


def test_a_valid_zero_is_not_an_unavailable_metric(tmp_path):
    """Zero spend, zero installs, zero D1-active or zero revenue on a day are real observations; only missing/impossible cells invalidate."""
    env = build_env(tmp_path, target=0.05)
    freeze_and_apply(env)
    release_all(env)
    ec, plan = env["ec"], env["plan"]
    obs = E.load_observations(ec)
    quiet = obs.copy()
    first = quiet.index[quiet.phase == "treatment"][0]
    quiet.loc[first, ["spend", "installs", "d1_active", "d30_net_revenue"]] = [0.0, 0, 0, 0.0]  # a unit-day with nothing at all
    out = R.run_readout(plan, quiet, E.clock(ec), POLICY, FORECAST, EXPERIMENT)
    assert out["input_problems"] == {} and out["invalid_readout"] is False and out["guardrails"]["cpi_ratio"] is not None
    no_installs_one_arm = obs.copy()  # but a group with NO installs at all has an unavailable CPI/retention: invalid, not zero
    no_installs_one_arm.loc[no_installs_one_arm.phase == "treatment", ["installs", "d1_active"]] = 0
    broken = R.run_readout(plan, no_installs_one_arm, E.clock(ec), POLICY, FORECAST, EXPERIMENT)
    assert broken["invalid_readout"] is True and broken["decision"] == "rollback"


def test_an_invalid_readout_is_executed_as_a_rollback_never_a_keep(tmp_path, monkeypatch):
    env = build_env(tmp_path, target=0.05)  # honest data would KEEP
    freeze_and_apply(env)
    release_all(env)
    assert do_readout(env)["decision"] == "keep"  # control: the unbroken data keeps
    broken = break_cells(E.load_observations(env["ec"]), "d1_active", np.nan, how_many=24 * 14)  # D1 retention unavailable for every treatment unit-day
    monkeypatch.setattr(E, "load_observations", lambda conn, as_of_clock=None: broken)
    assert env["led"].state()["budgets"]["meta"] == env["proposed"]
    res = execute(env)
    assert res["status"] == "rolled_back" and res["readout"]["invalid_readout"] is True and res["readout"]["decision_code"] == "rollback:invalid_readout_statistic"
    assert env["led"].state()["budgets"]["meta"] == env["baseline"]  # rolled back, not kept
    assert env["ec"].execute("SELECT outcome FROM experiment_outcomes").fetchone()[0] == "rolled_back"
