"""Sandbox ledger safeguards: ceiling/reserve, approval binding, transactional apply, stale approval, duplicate application,
hard constraints that approval cannot override, increase plan gate, rollback conflicts and blocked rollback, audit trail.

IMPORTANT: these tests exercise LEDGER MECHANICS. Increases are applied either through the `mechanics_test` qualification (recorded as
MECHANICS_TEST_APPLY) or through the REAL plan gate with a structurally valid plan whose power numbers are FABRICATED for speed
(tests/experiment_helpers.manual_plan). Neither says anything about statistical performance; the full simulated lifecycle with a real
planning study is in tests/test_experiment_lifecycle.py and the Phase 4 report."""
import copy
import json
import sqlite3

import pytest

from src.experiments import design as D
from src.experiments.plan import add_days, adequacy_from_numbers, plan_hash, validate_plan_structure
from src.ledger.ledger import Ledger
from src.ledger.proposal import Proposal, ProposalItem, make_proposal_id, proposal_hash
from src.policy.constraints import load_forecast_config, load_policy
from tests.experiment_helpers import manual_plan, refingerprint

CFG = load_policy()
FC = load_forecast_config()
BUDGETS = {"meta": 90.0, "google_uac": 80.0, "applovin": 100.0, "unity": 60.0, "tiktok": 70.0}  # total 400
ARBITRARY_HASH = "ab" * 32  # a well-formed 64-hex string that is the hash of NO plan


def new_ledger(tmp_path, name="l.db", mechanics=False, ceiling=None, cfg=CFG, budgets=BUDGETS):
    led = Ledger(tmp_path / name, cfg, FC, allow_mechanics_test=mechanics)
    led.initialize(budgets, ceiling, now="2025-09-01")
    return led


def propose(led, changes, qualification=None, plan=None, now="2025-09-08", source="rules_only"):
    """Build and store a proposal from {channel: new_budget}. `qualification` applies to increase items."""
    st = led.state()
    items = []
    for ch, cur in st["budgets"].items():
        new = changes.get(ch, cur)
        act = "increase_test" if new > cur else "decrease" if new < cur else "hold"
        q = qualification if (act == "increase_test" and qualification) else "none"
        items.append(ProposalItem(channel=ch, action=act, current_budget=cur, new_budget=new, qualification=q,
                                  experiment_plan_hash=plan if act == "increase_test" else None, reason="test"))
    pid = make_proposal_id("2025-09-08", items, st["version"], led.policy_hash) + f"-{len(led.audit())}"
    p = Proposal(proposal_id=pid, as_of="2025-09-08", ledger_version=st["version"], policy_hash=led.policy_hash, source=source, items=items)
    led.create_proposal(p, now)
    return pid


def approve_and_apply(led, changes, **kw):
    pid = propose(led, changes, **kw)
    a = led.approve(pid, "reviewer", "2025-09-08")
    assert a["ok"], a
    return pid, led.apply(pid, "2025-09-08")


# ---------------------------------------------------------------- ceiling, reserve, initialisation


def test_initial_state_has_a_ceiling_with_an_unallocated_reserve(tmp_path):
    led = new_ledger(tmp_path)
    s = led.state()
    assert s["version"] == 1 and s["kind"] == "INIT" and s["total"] == pytest.approx(400.0)
    assert s["ceiling"] == pytest.approx(480.0) and s["reserve"] == pytest.approx(80.0)  # 400 x (1 + 20%)
    assert all(c["channel_version"] == 1 and c["change_id"] == "INIT" for c in s["channels"].values())
    with pytest.raises(ValueError):
        Ledger(tmp_path / "bad.db", CFG, FC).initialize({"a": 100.0, "b": 10.0}, ceiling=200.0)  # a is 50% of the ceiling > 40%
    with pytest.raises(ValueError):
        Ledger(tmp_path / "bad2.db", CFG, FC).initialize({"a": 100.0, "b": 100.0, "c": 100.0}, ceiling=250.0)  # total above ceiling


def test_proposal_must_be_built_on_the_current_ledger_version(tmp_path):
    led = new_ledger(tmp_path)
    st = led.state()
    p = Proposal(proposal_id="P-x", as_of="2025-09-08", ledger_version=99, policy_hash=led.policy_hash,
                 items=[ProposalItem(channel="meta", action="hold", current_budget=90.0, new_budget=90.0, reason="r")])
    with pytest.raises(ValueError, match="ledger version"):
        led.create_proposal(p)
    assert st["version"] == 1


# ---------------------------------------------------------------- approval binding and successful apply


def test_approval_is_bound_to_proposal_hash_ledger_version_policy_and_plans_then_apply_changes_totals_and_reserve(tmp_path):
    led = new_ledger(tmp_path)
    before = led.state()
    pid = propose(led, {"meta": 81.0, "applovin": 90.0})  # two decreases free 19 of reserve
    a = led.approve(pid, "reviewer", "2025-09-08")
    row = led.conn.execute("SELECT proposal_hash, ledger_version, policy_hash, plan_hashes_json, approver FROM approvals WHERE proposal_id = ?", (pid,)).fetchone()
    body = json.loads(led.conn.execute("SELECT body_json FROM proposals WHERE proposal_id = ?", (pid,)).fetchone()[0])
    assert a["ok"] and row == (proposal_hash(body), 1, led.policy_hash, "[]", "reviewer")
    r = led.apply(pid, "2025-09-08")
    assert r["ok"] and r["kind"] == "APPLY" and r["version"] == 2
    after = led.state()
    assert before["total"] == pytest.approx(400.0) and before["reserve"] == pytest.approx(80.0)
    assert after["budgets"]["meta"] == 81.0 and after["budgets"]["applovin"] == 90.0 and after["budgets"]["unity"] == 60.0
    assert after["total"] == pytest.approx(381.0) and after["reserve"] == pytest.approx(99.0) and after["ceiling"] == before["ceiling"]
    assert after["channels"]["meta"]["channel_version"] == 2 and after["channels"]["meta"]["change_id"] == r["change_id"]
    assert after["channels"]["unity"]["channel_version"] == 1 and after["channels"]["unity"]["change_id"] == "INIT"  # untouched
    assert led.state(1) == before  # history is preserved
    assert led.conn.execute("SELECT status FROM proposals WHERE proposal_id = ?", (pid,)).fetchone()[0] == "applied"


def test_a_proposal_that_changes_nothing_is_rejected(tmp_path):
    led = new_ledger(tmp_path)
    pid = propose(led, {})
    assert led.approve(pid, "reviewer")["code"] == "no_budget_change"


# ---------------------------------------------------------------- duplicate application


def test_duplicate_application_is_prevented(tmp_path):
    led = new_ledger(tmp_path)
    pid, first = approve_and_apply(led, {"meta": 81.0})
    assert first["ok"]
    again = led.apply(pid, "2025-09-09")
    assert not again["ok"] and again["code"] == "already_applied"
    assert led.head() == 2 and led.state()["budgets"]["meta"] == 81.0  # nothing applied twice
    with pytest.raises(sqlite3.IntegrityError):  # the database itself refuses a second application row
        led.conn.execute("INSERT INTO applied_changes VALUES ('C-dup', 'APPLY', ?, NULL, 'APPLY', 1, 2, '{}', 'active', NULL, 'x', 'y')", (pid,))


# ---------------------------------------------------------------- stale approval


def test_stale_approval_is_rejected_when_the_ledger_moved_on(tmp_path):
    led = new_ledger(tmp_path)
    p1 = propose(led, {"meta": 81.0})
    p2 = propose(led, {"unity": 54.0})  # both built and approved against version 1
    assert led.approve(p1, "reviewer")["ok"] and led.approve(p2, "reviewer")["ok"]
    assert led.apply(p1)["ok"]  # head is now 2
    r = led.apply(p2)
    assert not r["ok"] and r["code"] == "stale_ledger_version" and r["detail"] == {"approved_on": 1, "head": 2}
    assert led.head() == 2 and led.state()["budgets"]["unity"] == 60.0
    p3 = propose(led, {"tiktok": 63.0})  # built on version 2
    assert led.apply(p3)["code"] == "not_approved"  # never approved
    stale_open = propose(led, {"google_uac": 72.0})
    led2 = propose(led, {"meta": 74.0})
    assert led.approve(led2, "reviewer")["ok"] and led.apply(led2)["ok"]  # head 3
    refused = led.approve(stale_open, "reviewer")  # approving a proposal built on an old version is refused too
    assert not refused["ok"] and refused["code"] == "stale_ledger_version"


def test_changed_policy_configuration_makes_an_approval_stale(tmp_path):
    led = new_ledger(tmp_path)
    pid = propose(led, {"meta": 81.0})
    assert led.approve(pid, "reviewer")["ok"]
    changed = copy.deepcopy(CFG)
    changed["constraints"]["per_channel_change_cap"] = 0.25  # any policy change alters the hash
    led2 = Ledger(tmp_path / "l.db", changed, FC)
    r = led2.apply(pid)
    assert not r["ok"] and r["code"] == "stale_policy_config" and led2.head() == 1


def test_tampering_with_an_approved_proposal_is_detected(tmp_path):
    led = new_ledger(tmp_path)
    pid = propose(led, {"meta": 81.0})
    assert led.approve(pid, "reviewer")["ok"]
    body = json.loads(led.conn.execute("SELECT body_json FROM proposals WHERE proposal_id = ?", (pid,)).fetchone()[0])
    body["items"][0]["new_budget"] = 85.0 if body["items"][0]["channel"] == "applovin" else body["items"][0]["new_budget"]
    for it in body["items"]:
        if it["channel"] == "meta":
            it["new_budget"] = 72.0  # a different content after approval
    led.conn.execute("UPDATE proposals SET body_json = ? WHERE proposal_id = ?", (json.dumps(body), pid))
    r = led.apply(pid)
    assert not r["ok"] and r["code"] == "proposal_hash_mismatch" and led.head() == 1


# ---------------------------------------------------------------- hard constraints beat approval


def test_approval_refuses_proposals_that_break_the_change_cap(tmp_path):
    led = new_ledger(tmp_path, mechanics=True)
    pid = propose(led, {"meta": 112.0}, qualification="mechanics_test")  # +24.4% > 20%
    r = led.approve(pid, "reviewer")
    assert not r["ok"] and r["code"] == "constraint_violation"
    assert {v["constraint"] for v in r["detail"]["violations"]} == {"per_channel_change_cap"}
    assert led.audit("approval_refused")[-1]["details"]["code"] == "constraint_violation"


def test_approval_refuses_proposals_that_break_the_channel_share_limit(tmp_path):
    # ceiling 240: a channel may hold at most 96 (40%). 'a' starts at 95 and a +5.3% step would take it to 100 (41.7%); the total stays under the ceiling
    led = new_ledger(tmp_path, mechanics=True, ceiling=240.0, budgets={"a": 95.0, "b": 30.0, "c": 30.0, "d": 30.0})
    pid = propose(led, {"a": 100.0}, qualification="mechanics_test")
    r = led.approve(pid, "reviewer")
    assert not r["ok"] and {v["constraint"] for v in r["detail"]["violations"]} == {"max_channel_share"}


def test_ceiling_floor_and_negative_budgets_are_refused(tmp_path):
    led = new_ledger(tmp_path, mechanics=True, ceiling=410.0)  # reserve only 10
    over = propose(led, {"meta": 99.0, "google_uac": 88.0}, qualification="mechanics_test")  # +9 +8 = 17 > 10
    r = led.approve(over, "reviewer")
    assert r["code"] == "constraint_violation" and "ceiling" in {v["constraint"] for v in r["detail"]["violations"]}
    small = Ledger(tmp_path / "s.db", CFG, FC)
    small.initialize({"a": 11.0, "b": 12.0, "c": 13.0, "d": 14.0}, ceiling=60.0)
    low = propose(small, {"a": 9.5})  # -13.6% is inside the cap but below the 10.0 floor
    assert "min_spend_floor" in {v["constraint"] for v in small.approve(low, "reviewer")["detail"]["violations"]}
    with pytest.raises(ValueError):  # a negative budget cannot even be built as a proposal item
        ProposalItem(channel="a", action="decrease", current_budget=11.0, new_budget=-1.0, reason="x")


def test_apply_rechecks_constraints_even_if_an_approval_row_is_forced_in(tmp_path):
    """Hard constraints beat humans: an 'approval' inserted directly (bypassing approve()) still cannot get a violation applied."""
    led = new_ledger(tmp_path, mechanics=True)
    pid = propose(led, {"meta": 112.0}, qualification="mechanics_test")  # +24%: breaks the cap
    h = led.conn.execute("SELECT proposal_hash FROM proposals WHERE proposal_id = ?", (pid,)).fetchone()[0]
    led.conn.execute("INSERT INTO approvals (proposal_id, approver, decision, proposal_hash, ledger_version, policy_hash, plan_hashes_json, decided_at) "
                     "VALUES (?, 'someone with authority', 'approve', ?, 1, ?, '[]', 'now')", (pid, h, led.policy_hash))
    led.conn.execute("UPDATE proposals SET status = 'approved' WHERE proposal_id = ?", (pid,))
    r = led.apply(pid)
    assert not r["ok"] and r["code"] == "constraint_violation" and led.head() == 1
    # and a raw stored body with a NEGATIVE budget (bypassing the pydantic model) is refused the same way
    body = json.loads(led.conn.execute("SELECT body_json FROM proposals WHERE proposal_id = ?", (pid,)).fetchone()[0])
    for it in body["items"]:
        if it["channel"] == "meta":
            it["new_budget"], it["action"] = -5.0, "decrease"
    pid2 = "P-raw-negative"
    h2 = proposal_hash(body)
    led.conn.execute("INSERT INTO proposals VALUES (?,?,?,?,?,?,'approved','now')", (pid2, h2, json.dumps(body), 1, led.policy_hash, "2025-09-08"))
    led.conn.execute("INSERT INTO approvals (proposal_id, approver, decision, proposal_hash, ledger_version, policy_hash, plan_hashes_json, decided_at) "
                     "VALUES (?, 'x', 'approve', ?, 1, ?, '[]', 'now')", (pid2, h2, led.policy_hash))
    r2 = led.apply(pid2)
    assert r2["code"] == "constraint_violation" and "non_negative" in {v["constraint"] for v in r2["detail"]["violations"]}


def test_action_must_match_the_budget_change(tmp_path):
    led = new_ledger(tmp_path)
    pid = propose(led, {"meta": 81.0})
    body = json.loads(led.conn.execute("SELECT body_json FROM proposals WHERE proposal_id = ?", (pid,)).fetchone()[0])
    for it in body["items"]:
        if it["channel"] == "meta":
            it["action"] = "hold"  # says hold but moves the budget
    h = proposal_hash(body)
    led.conn.execute("UPDATE proposals SET body_json = ?, proposal_hash = ? WHERE proposal_id = ?", (json.dumps(body), h, pid))
    assert led.approve(pid, "reviewer")["code"] == "invalid_action_budget"


# ---------------------------------------------------------------- increase gate (mechanics, not a qualified experiment)


def test_increase_needs_a_frozen_plan_and_policy_style_plan_required_proposals_cannot_be_applied(tmp_path):
    led = new_ledger(tmp_path)
    pid = propose(led, {"meta": 99.0}, qualification="plan_required")  # exactly what the Phase 3 policy emits for an increase test
    r = led.approve(pid, "reviewer")
    assert not r["ok"] and r["code"] == "increase_requires_frozen_plan"
    assert led.head() == 1  # a Phase 3 increase test CANNOT be executed: no experiment plan exists yet


def real_plan(led, **kw):
    """A structurally valid plan for meta 90 -> 99 on 2025-09-08 under this ledger's configuration (power numbers fabricated; see experiment_helpers)."""
    kw.setdefault("policy_hash", led.policy_hash)
    return manual_plan(baseline=90.0, proposed=99.0, as_of="2025-09-08", **kw)


def test_an_arbitrary_hash_can_no_longer_be_registered_or_qualify_an_increase(tmp_path):
    led = new_ledger(tmp_path)
    for junk in (ARBITRARY_HASH, "not-a-hash", {"channel": "meta"}):
        with pytest.raises(ValueError, match="structurally valid"):
            led.register_frozen_plan(junk, "test", "2025-09-08")  # the ledger stores CONTENT and recomputes the hash itself
    assert led.conn.execute("SELECT COUNT(*) FROM frozen_plans").fetchone()[0] == 0
    unregistered = propose(led, {"meta": 99.0}, qualification="experiment_backed", plan=ARBITRARY_HASH)
    r = led.approve(unregistered, "reviewer")
    assert not r["ok"] and r["code"] == "frozen_plan_not_registered" and led.head() == 1
    missing = propose(led, {"meta": 99.0}, qualification="experiment_backed", plan=None)
    assert led.approve(missing, "reviewer")["code"] == "increase_requires_frozen_plan"  # experiment_backed with no plan at all


def test_a_real_registered_plan_qualifies_the_increase_and_is_part_of_the_approval(tmp_path):
    led = new_ledger(tmp_path)
    plan = real_plan(led)
    h = led.register_frozen_plan(plan, "designer", "2025-09-08")
    assert h == plan_hash(plan) and led.get_plan(h) == plan  # full canonical content stored under the recomputed hash
    ok = propose(led, {"meta": 99.0}, qualification="experiment_backed", plan=h)
    assert led.approve(ok, "reviewer")["ok"]
    approvals = led.conn.execute("SELECT plan_hashes_json FROM approvals WHERE proposal_id = ?", (ok,)).fetchone()[0]
    assert json.loads(approvals) == [h]  # the plan hash is part of what was approved
    r = led.apply(ok)
    assert r["ok"] and r["kind"] == "APPLY" and led.state()["budgets"]["meta"] == 99.0  # a plan-backed increase, not MECHANICS_TEST_APPLY
    assert led.audit("plan_registered")[0]["details"]["plan_hash"] == h
    with pytest.raises(sqlite3.IntegrityError):
        led.register_frozen_plan(plan, "designer", "2025-09-08")  # the same plan cannot be registered twice


def test_an_altered_plan_is_rejected_at_approval_and_again_at_application(tmp_path):
    led = new_ledger(tmp_path)
    plan = real_plan(led)
    h = led.register_frozen_plan(plan, "designer", "2025-09-08")
    led.conn.execute("DROP TRIGGER frozen_plans_no_update")  # simulate tampering with storage (normally blocked)
    tampered = copy.deepcopy(plan)
    tampered["decision_rule"]["target_d30_roas"] = 0.1
    tampered["target"]["target_d30_roas"] = 0.1
    led.conn.execute("UPDATE frozen_plans SET plan_json = ? WHERE plan_hash = ?", (json.dumps(tampered), h))
    pid = propose(led, {"meta": 99.0}, qualification="experiment_backed", plan=h)
    r = led.approve(pid, "reviewer")
    assert not r["ok"] and r["code"] == "plan_hash_mismatch" and r["detail"]["recomputed"] == plan_hash(tampered) != h
    # approved honestly first, tampered afterwards, then applied: still rejected at application
    led2 = new_ledger(tmp_path, "l2.db")
    plan2 = real_plan(led2)
    h2 = led2.register_frozen_plan(plan2, "designer", "2025-09-08")
    pid2 = propose(led2, {"meta": 99.0}, qualification="experiment_backed", plan=h2)
    assert led2.approve(pid2, "reviewer")["ok"]
    led2.conn.execute("DROP TRIGGER frozen_plans_no_update")
    led2.conn.execute("UPDATE frozen_plans SET plan_json = ? WHERE plan_hash = ?", (json.dumps(tampered), h2))
    r2 = led2.apply(pid2)
    assert not r2["ok"] and r2["code"] == "plan_hash_mismatch" and led2.head() == 1 and led2.state()["budgets"]["meta"] == 90.0
    assert led2.audit("apply_rejected")[-1]["details"]["code"] == "plan_hash_mismatch"


@pytest.mark.parametrize("what,kwargs,item_changes,needle", [
    ("another channel's plan", {"channel": "tiktok", "baseline": 70.0, "proposed": 77.0}, {}, "channel"),
    ("a different exact step", {"proposed": 98.0}, {}, "proposed budget"),
    ("a different baseline", {"baseline": 80.0, "proposed": 99.0}, {}, "baseline budget"),
    ("another policy configuration", {"policy_hash": "e" * 64}, {}, "different policy configuration"),
    ("another decision date", {"as_of": "2025-09-15"}, {}, "decision date"),
])
def test_a_plan_for_something_else_is_rejected_with_the_specific_mismatch(tmp_path, what, kwargs, item_changes, needle):
    led = new_ledger(tmp_path)
    args = {"baseline": 90.0, "proposed": 99.0, "as_of": "2025-09-08", "policy_hash": led.policy_hash, **kwargs}
    plan = manual_plan(**args)
    h = led.register_frozen_plan(plan, "designer", "2025-09-08")
    pid = propose(led, {"meta": 99.0}, qualification="experiment_backed", plan=h)
    r = led.approve(pid, "reviewer")
    assert not r["ok"] and r["code"] == "plan_binding_mismatch", what
    assert any(needle in e for e in r["detail"]["errors"]), r["detail"]["errors"]
    assert led.head() == 1


def test_an_inadequate_plan_can_be_recorded_but_never_qualifies_an_increase(tmp_path):
    led = new_ledger(tmp_path)
    plan = real_plan(led, adequate=False)
    h = led.register_frozen_plan(plan, "designer", "2025-09-08")  # a fact, stored
    pid = propose(led, {"meta": 99.0}, qualification="experiment_backed", plan=h)
    r = led.approve(pid, "reviewer")
    assert not r["ok"] and r["code"] == "plan_inadequate" and "adequacy standards" in r["detail"]["reasons"][0]
    # even a forced-in approval cannot get it applied
    force_approve(led, pid)
    r2 = led.apply(pid)
    assert r2["code"] == "plan_inadequate" and led.head() == 1


def test_a_hash_consistent_but_structurally_broken_plan_is_rejected_by_the_ledger(tmp_path):
    """Bypass register_frozen_plan and store a plan whose treated budgets do NOT add up to the step, under its own correct hash."""
    led = new_ledger(tmp_path)
    plan = real_plan(led)
    first_treated = next(u for u in plan["units"] if u["arm"] == "treatment")
    first_treated["planned_incremental_spend"] += 0.5
    h = plan_hash(plan)
    led.conn.execute("INSERT INTO frozen_plans VALUES (?,?,?,?)", (h, json.dumps(plan), "bypass", "now"))
    pid = propose(led, {"meta": 99.0}, qualification="experiment_backed", plan=h)
    r = led.approve(pid, "reviewer")
    assert r["code"] == "plan_invalid" and any("add up to the exact channel-level step" in e for e in r["detail"]["errors"])


def store_bypassing_registration(led, plan):
    """Put a plan straight into frozen_plans under its own correct hash (what a tampering party with storage access could do)."""
    h = plan_hash(plan)
    led.conn.execute("INSERT INTO frozen_plans VALUES (?,?,?,?)", (h, json.dumps(plan), "bypass", "now"))
    return h


def weakened_plan(led):
    """REGRESSION (independent review): a plan whose real power at 1.5x target is 30% but which LOWERED the required power from 0.80 to 0.01
    while keeping the original experiment_config_hash. Its recorded adequacy is internally consistent (adequate = True)."""
    plan = real_plan(led, adequate=False)  # fabricated numbers: 30% power at 1.5x target in every window
    pa = plan["power_assessment"]
    assert pa["per_duration"]["14"]["alternatives"][4]["keep_rate"] == 0.30 and pa["adequate"] is False
    pa["standards"]["target_power"] = 0.01
    pa["chosen_treatment_days"] = 14
    pa["adequate"], pa["reasons"] = adequacy_from_numbers(pa)
    assert pa["adequate"] is True  # consistent with its own (weakened) standard
    return plan


def test_a_plan_that_lowered_the_required_power_is_refused_under_the_original_configuration_hash(tmp_path):
    led = new_ledger(tmp_path)
    plan = weakened_plan(led)
    assert validate_plan_structure(plan) == []  # structurally fine, hash unchanged, self-consistent: only the authoritative configuration can catch it
    assert plan["experiment_config_hash"] == led.experiment_config_hash
    with pytest.raises(ValueError, match="does not follow the ledger's experiment configuration"):
        led.register_frozen_plan(plan, "designer", "2025-09-08")
    assert led.conn.execute("SELECT COUNT(*) FROM frozen_plans").fetchone()[0] == 0
    h = store_bypassing_registration(led, plan)  # even if it reaches storage some other way ...
    pid = propose(led, {"meta": 99.0}, qualification="experiment_backed", plan=h)
    r = led.approve(pid, "reviewer")
    assert not r["ok"] and r["code"] == "plan_config_violation" and any("adequacy standards differ" in e for e in r["detail"]["errors"])
    force_approve(led, pid)  # ... and even with a forced approval row, application re-checks and refuses
    r2 = led.apply(pid)
    assert r2["code"] == "plan_config_violation" and led.head() == 1 and led.state()["budgets"]["meta"] == 90.0


def altered_plan(mutate, **kw):
    """A plan that keeps the ORIGINAL experiment_config_hash but changes one setting; the design fingerprint is refreshed so only that defect remains."""
    plan = manual_plan(baseline=90.0, proposed=99.0, as_of="2025-09-08", **kw)
    mutate(plan)
    return refingerprint(plan)


def change_horizon(p):
    """A 14-day outcome horizon, with the readout date moved consistently so the plan stays structurally valid."""
    p["windows"]["outcome_horizon_days"] = 14
    p["windows"]["readout_available_on"] = add_days(p["windows"]["treatment_end"], 14 + p["windows"]["reporting_lag_days"])


@pytest.mark.parametrize("name,mutate,needle", [
    ("looser alpha", lambda p: p["estimator"].__setitem__("alpha", 0.30), "estimator settings differ"),
    ("wider delivery band", lambda p: p["estimator"].__setitem__("min_delivery_fraction", 0.0), "estimator settings differ"),
    ("looser CPI guardrail", lambda p: p["guardrails"].__setitem__("max_cpi_ratio", 9.0), "guardrail settings differ"),
    ("looser D1 guardrail", lambda p: p["guardrails"].__setitem__("min_d1_retention_change", -0.9), "guardrail settings differ"),
    ("fewer allowed power replicates", lambda p: p["power_assessment"].__setitem__("replicates", 100), "replicates/seed differ"),
    ("a different horizon", change_horizon, "supported D30"),
    ("weaker false-keep ceiling", lambda p: p["power_assessment"]["standards"].__setitem__("max_false_keep", 0.9), "adequacy standards differ"),
    ("fewer units per arm", lambda p: p["power_assessment"]["standards"].__setitem__("min_units_per_arm", 2), "adequacy standards differ"),
])
def test_altered_settings_under_an_unchanged_configuration_hash_are_refused_by_the_ledger(tmp_path, name, mutate, needle):
    led = new_ledger(tmp_path)
    plan = altered_plan(mutate)
    assert plan["experiment_config_hash"] == led.experiment_config_hash  # the declared configuration hash is the ORIGINAL one
    assert validate_plan_structure(plan) == []  # structurally fine: only the authoritative configuration can refuse it
    with pytest.raises(ValueError, match="does not follow the ledger's experiment configuration") as exc:
        led.register_frozen_plan(plan, "designer", "2025-09-08")
    assert needle in str(exc.value), name


def test_a_plan_frozen_under_a_different_experiment_configuration_needs_that_configuration_loaded(tmp_path):
    led = new_ledger(tmp_path)
    lowered = D.load_config()
    lowered["power"]["target_power"] = 0.01
    plan = weakened_plan(led)
    plan["experiment_config_hash"] = D.config_hash(lowered)  # self-consistent under ITS OWN (weakened) configuration
    with pytest.raises(ValueError, match="load the plan's configuration first"):
        led.register_frozen_plan(plan, "designer", "2025-09-08")  # the ledger's authoritative configuration is not that one
    h = store_bypassing_registration(led, plan)
    pid = propose(led, {"meta": 99.0}, qualification="experiment_backed", plan=h)
    r = led.approve(pid, "reviewer")
    assert r["code"] == "experiment_config_not_loaded" and "reload_config" in r["detail"]["action"] and led.head() == 1
    # The authoritative configuration changes only through the audited reload; the snapshot is a copy, not the caller's dict
    lowered["power"]["target_power"] = 0.5
    assert led.experiment_cfg["power"]["target_power"] == 0.80
    led.reload_config(CFG, FC, "2025-09-09", experiment_cfg=D.load_config())
    assert led.audit("config_reloaded")[-1]["details"]["new_experiment_config_hash"] == led.experiment_config_hash


def test_planning_numbers_recorded_for_another_design_or_with_forged_errors_are_refused(tmp_path):
    led = new_ledger(tmp_path)
    led.register_frozen_plan(real_plan(led), "designer", "2025-09-08")  # control: the honest fixture is accepted
    other = real_plan(led)
    other["target"]["target_d30_roas"] = other["decision_rule"]["target_d30_roas"] = 0.4  # the numbers were recorded for target 0.5
    with pytest.raises(ValueError, match="does not follow"):
        led.register_frozen_plan(other, "designer", "2025-09-08")
    inflated = real_plan(led, adequate=False)
    for a in inflated["power_assessment"]["per_duration"]["14"]["alternatives"]:
        a["mc_se"] = 0.5  # a huge MC SE would widen the false-keep tolerance
    with pytest.raises(ValueError, match="Monte Carlo SE"):
        led.register_frozen_plan(inflated, "designer", "2025-09-08")


def test_mechanics_test_increases_need_the_flag_and_are_labelled_in_the_ledger(tmp_path):
    off = new_ledger(tmp_path, "off.db", mechanics=False)
    pid = propose(off, {"meta": 99.0}, qualification="mechanics_test")
    assert off.approve(pid, "reviewer")["code"] == "mechanics_test_not_enabled"
    on = new_ledger(tmp_path, "on.db", mechanics=True)
    pid2 = propose(on, {"meta": 99.0}, qualification="mechanics_test")
    assert on.approve(pid2, "reviewer")["ok"]
    r = on.apply(pid2)
    assert r["ok"] and r["kind"] == "MECHANICS_TEST_APPLY"
    note = on.conn.execute("SELECT kind, note FROM ledger_versions WHERE version = 2").fetchone()
    assert note[0] == "MECHANICS_TEST_APPLY" and "not an experiment-backed increase" in note[1]


def test_a_decrease_needs_no_experiment_plan(tmp_path):
    led = new_ledger(tmp_path)
    _, r = approve_and_apply(led, {"unity": 54.0})
    assert r["ok"] and led.state()["budgets"]["unity"] == 54.0


# ---------------------------------------------------------------- atomicity


def test_apply_is_all_or_nothing(tmp_path):
    led = new_ledger(tmp_path)
    pid = propose(led, {"meta": 81.0, "unity": 54.0})
    assert led.approve(pid, "reviewer")["ok"]
    before = led.state()
    led.conn.execute("CREATE TRIGGER boom BEFORE INSERT ON applied_changes BEGIN SELECT RAISE(ABORT, 'simulated failure'); END")
    with pytest.raises(sqlite3.IntegrityError, match="simulated failure"):
        led.apply(pid)
    assert led.head() == 1 and led.state() == before  # no new version, no partial budgets
    assert led.conn.execute("SELECT status FROM proposals WHERE proposal_id = ?", (pid,)).fetchone()[0] == "approved"
    assert led.conn.execute("SELECT COUNT(*) FROM applied_changes").fetchone()[0] == 0
    assert led.audit("apply_failed")  # the failure itself is audited
    led.conn.execute("DROP TRIGGER boom")
    assert led.apply(pid)["ok"]  # and the proposal can still be applied once the fault is gone


# ---------------------------------------------------------------- rollback


def test_rollback_restores_budgets_and_reserve_and_cannot_be_repeated(tmp_path):
    led = new_ledger(tmp_path)
    _, applied = approve_and_apply(led, {"meta": 81.0, "unity": 54.0})
    mid = led.state()
    r = led.rollback(applied["change_id"], "reviewer", "test rollback", "2025-09-09")
    assert r["ok"] and r["status"] == "rolled_back" and r["version"] == 3
    after = led.state()
    assert after["budgets"] == BUDGETS and after["reserve"] == pytest.approx(80.0) and mid["reserve"] == pytest.approx(95.0)
    assert after["channels"]["meta"]["channel_version"] == 3  # the counter only goes up
    assert after["channels"]["meta"]["change_id"] == "INIT"  # the restored value is the initial one, so its provenance id is back on the row
    assert led.conn.execute("SELECT status, rolled_back_by FROM applied_changes WHERE change_id = ?", (applied["change_id"],)).fetchone() == ("rolled_back", r["change_id"])
    assert led.conn.execute("SELECT kind, change_id, note FROM ledger_versions WHERE version = 3").fetchone() == ("ROLLBACK", r["change_id"], "rollback of " + applied["change_id"] + ": test rollback")
    assert led.rollback(applied["change_id"], "reviewer", "again")["code"] == "already_rolled_back"
    assert led.rollback(r["change_id"], "reviewer", "undo the undo")["code"] == "cannot_roll_back_a_rollback"
    assert led.rollback("C-nope", "reviewer", "x")["code"] == "change_not_found"
    assert led.head() == 3  # none of the refused attempts created a version


def test_rollback_is_blocked_when_a_later_change_touched_the_same_channel(tmp_path):
    led = new_ledger(tmp_path)
    _, a = approve_and_apply(led, {"meta": 85.0})
    _, b = approve_and_apply(led, {"meta": 80.0})  # a later change on the same channel
    r = led.rollback(a["change_id"], "reviewer", "revert A")
    assert r["status"] == "rollback_blocked" and r["code"] == "channel_changed_since"
    conflict = r["detail"]["conflicts"]["meta"]
    assert conflict["current_change_id"] == b["change_id"] and conflict["expected_change_id"] == a["change_id"] and conflict["current_budget"] == 80.0
    assert led.state()["budgets"]["meta"] == 80.0 and led.head() == 3
    assert led.audit("rollback_blocked")[-1]["details"]["code"] == "channel_changed_since"
    assert led.rollback(b["change_id"], "reviewer", "revert B first")["ok"]  # unwinding in order works
    assert led.rollback(a["change_id"], "reviewer", "now revert A")["ok"] and led.state()["budgets"]["meta"] == 90.0


def test_rollback_of_a_multi_channel_change_is_all_or_nothing(tmp_path):
    led = new_ledger(tmp_path)
    _, a = approve_and_apply(led, {"meta": 81.0, "google_uac": 72.0})
    _, b = approve_and_apply(led, {"google_uac": 70.0})  # later change touches only one of A's channels
    r = led.rollback(a["change_id"], "reviewer", "revert A")
    assert r["status"] == "rollback_blocked" and set(r["detail"]["conflicts"]) == {"google_uac"}
    assert led.state()["budgets"]["meta"] == 81.0  # meta was NOT restored on its own: no partial rollback


def test_rollback_is_blocked_when_restoring_would_break_the_ceiling(tmp_path):
    led = new_ledger(tmp_path, mechanics=True, ceiling=405.0)  # tight: reserve 5
    _, cut = approve_and_apply(led, {"meta": 81.0})  # frees 9 -> reserve 14
    for ch, new in (("unity", 66.0), ("tiktok", 77.0)):  # other channels use the freed reserve (mechanics tests, not experiments)
        pid = propose(led, {ch: new}, qualification="mechanics_test")
        assert led.approve(pid, "reviewer")["ok"] and led.apply(pid)["ok"]
    assert led.state()["reserve"] == pytest.approx(1.0)
    r = led.rollback(cut["change_id"], "reviewer", "restore meta")  # would need +9 but only 1 is unallocated
    assert r["status"] == "rollback_blocked" and r["code"] == "constraint_violation"
    assert "ceiling" in {v["constraint"] for v in r["detail"]["violations"]}
    assert led.state()["budgets"]["meta"] == 81.0


def force_approve(led, pid):
    """Insert an approval row directly (bypassing approve()) to prove apply re-checks everything itself."""
    h, body = led.conn.execute("SELECT proposal_hash, body_json FROM proposals WHERE proposal_id = ?", (pid,)).fetchone()
    plans = sorted({i["experiment_plan_hash"] for i in json.loads(body)["items"] if i.get("experiment_plan_hash")})
    head = led.head()
    led.conn.execute("INSERT INTO approvals (proposal_id, approver, decision, proposal_hash, ledger_version, policy_hash, plan_hashes_json, decided_at) "
                     "VALUES (?, 'forced', 'approve', ?, ?, ?, ?, 'now')", (pid, h, head, led.policy_hash, json.dumps(plans)))
    led.conn.execute("UPDATE proposals SET status = 'approved' WHERE proposal_id = ?", (pid,))


# ---------------------------------------------------------------- immediate reversibility


def test_a_cut_whose_exact_inverse_breaks_the_cap_is_refused_at_approval_with_an_explicit_reason(tmp_path):
    led = new_ledger(tmp_path)
    pid = propose(led, {"meta": 72.0})  # 90 -> 72 is -20%: inside the cap; the inverse 72 -> 90 is +25%: outside it
    r = led.approve(pid, "reviewer")
    assert not r["ok"] and r["code"] == "not_reversible" and "could not be rolled back" in r["detail"]["detail"]
    inv = r["detail"]["inverse_violations"]
    assert [v["constraint"] for v in inv] == ["per_channel_change_cap"] and inv[0]["channel"] == "meta" and "+25.0%" in inv[0]["detail"]
    assert led.head() == 1 and led.state()["budgets"]["meta"] == 90.0
    assert led.audit("approval_refused")[-1]["details"]["code"] == "not_reversible"


def test_reversibility_is_checked_again_at_apply_even_if_an_approval_is_forced_in(tmp_path):
    led = new_ledger(tmp_path)
    pid = propose(led, {"meta": 72.0})
    force_approve(led, pid)
    r = led.apply(pid)
    assert not r["ok"] and r["code"] == "not_reversible" and led.head() == 1
    assert led.conn.execute("SELECT status FROM proposals WHERE proposal_id = ?", (pid,)).fetchone()[0] == "approved"  # left as it was
    assert led.audit("apply_rejected")[-1]["details"]["code"] == "not_reversible"


def test_reversible_cut_control_applies_and_rolls_back_and_the_boundary_is_exact(tmp_path):
    led = new_ledger(tmp_path)
    _, cut = approve_and_apply(led, {"meta": 75.0})  # -16.7%: the inverse is +20.0%, exactly at the cap
    assert cut["ok"]
    back = led.rollback(cut["change_id"], "reviewer", "restore", "2025-09-09")
    assert back["ok"] and led.state()["budgets"]["meta"] == 90.0  # the cap was not bypassed: +20% is allowed
    just_over = propose(led, {"meta": 74.9})  # inverse +20.16% > 20%
    assert led.approve(just_over, "reviewer")["code"] == "not_reversible"


def test_multi_channel_inverse_is_checked_as_a_whole(tmp_path):
    led = new_ledger(tmp_path)
    pid = propose(led, {"meta": 81.0, "unity": 48.0})  # unity 60 -> 48 is -20%; its inverse is +25%
    r = led.approve(pid, "reviewer")
    assert r["code"] == "not_reversible" and {v["channel"] for v in r["detail"]["inverse_violations"]} == {"unity"}


def test_runtime_rollback_check_still_blocks_when_the_configuration_has_since_tightened(tmp_path):
    led = new_ledger(tmp_path)
    _, cut = approve_and_apply(led, {"meta": 81.0})  # -10%: inverse +11.1%, fine under the 20% cap
    tighter = copy.deepcopy(CFG)
    tighter["constraints"]["per_channel_change_cap"] = 0.10
    led.reload_config(tighter, FC, "2025-09-09")
    r = led.rollback(cut["change_id"], "reviewer", "restore")
    assert r["status"] == "rollback_blocked" and "per_channel_change_cap" in {v["constraint"] for v in r["detail"]["violations"]}  # caps are never bypassed
    assert led.state()["budgets"]["meta"] == 81.0


# ---------------------------------------------------------------- configuration snapshot and approval hashes


def test_ledger_configuration_is_an_immutable_snapshot_not_a_shared_dictionary(tmp_path):
    mine = copy.deepcopy(CFG)
    led = Ledger(tmp_path / "l.db", mine, FC, allow_mechanics_test=True)
    led.initialize(BUDGETS, now="2025-09-01")
    h0 = led.policy_hash
    mine["constraints"]["per_channel_change_cap"] = 0.50  # the caller mutates ITS dictionary afterwards: no effect on the ledger
    mine["marginal"]["incremental_quality_haircut"] = 0.70
    assert led.policy_hash == h0 and led.cfg["constraints"]["per_channel_change_cap"] == 0.20
    led.cfg["constraints"]["per_channel_change_cap"] = 0.99  # mutating the returned copy changes nothing either
    assert led.cfg["constraints"]["per_channel_change_cap"] == 0.20 and led.policy_hash == h0
    pid = propose(led, {"meta": 112.0}, qualification="mechanics_test")  # +24%: still refused under the ledger's own 20% cap
    assert led.approve(pid, "reviewer")["code"] == "constraint_violation"


def live_ledger(tmp_path):
    live = {"policy": copy.deepcopy(CFG), "fc": copy.deepcopy(FC)}
    led = Ledger(tmp_path / "l.db", copy.deepcopy(CFG), copy.deepcopy(FC), active_config_provider=lambda: (live["policy"], live["fc"]))
    led.initialize(BUDGETS, now="2025-09-01")
    return led, live


def test_active_policy_change_between_approval_and_application_rejects_atomically_with_audit(tmp_path):
    """Reproducer from independent review: approve a valid decrease, change marginal.incremental_quality_haircut 0.80 -> 0.70, apply."""
    led, live = live_ledger(tmp_path)
    pid = propose(led, {"meta": 81.0})
    assert led.approve(pid, "reviewer")["ok"]
    live["policy"]["marginal"]["incremental_quality_haircut"] = 0.70  # the ACTIVE configuration changed; the ledger has not reloaded it
    r = led.apply(pid)
    assert not r["ok"] and r["code"] == "stale_policy_config"
    assert r["detail"]["ledger_snapshot_hash"] != r["detail"]["active_configuration_hash"] and "reload_config" in r["detail"]["action"]
    assert led.head() == 1 and led.state()["budgets"]["meta"] == 90.0 and led.conn.execute("SELECT COUNT(*) FROM applied_changes").fetchone()[0] == 0
    assert led.conn.execute("SELECT status FROM proposals WHERE proposal_id = ?", (pid,)).fetchone()[0] == "approved"
    ev = led.audit("apply_rejected")[-1]
    assert ev["details"]["code"] == "stale_policy_config" and ev["details"]["active_configuration_hash"] == r["detail"]["active_configuration_hash"]
    live["policy"]["marginal"]["incremental_quality_haircut"] = CFG["marginal"]["incremental_quality_haircut"]  # change reverted: the hash matches again
    assert led.apply(pid)["ok"]


def test_active_forecast_configuration_change_blocks_approval_and_application(tmp_path):
    led, live = live_ledger(tmp_path)
    p1 = propose(led, {"meta": 81.0})
    live["fc"]["quality"]["spike_ratio"] = 4.0  # forecast/quality configuration is part of the decision configuration
    refused = led.approve(p1, "reviewer")
    assert not refused["ok"] and refused["code"] == "stale_policy_config" and led.audit("approval_refused")[-1]["details"]["code"] == "stale_policy_config"
    live["fc"]["quality"]["spike_ratio"] = FC["quality"]["spike_ratio"]
    assert led.approve(p1, "reviewer")["ok"]
    live["fc"]["forecast"]["train_window_cohorts"] = 59
    assert led.apply(p1)["code"] == "stale_policy_config" and led.head() == 1


def test_reload_config_is_audited_and_makes_earlier_approvals_stale(tmp_path):
    led = new_ledger(tmp_path)
    pid = propose(led, {"meta": 81.0})
    assert led.approve(pid, "reviewer")["ok"]
    new_cfg = copy.deepcopy(CFG)
    new_cfg["marginal"]["incremental_quality_haircut"] = 0.70
    res = led.reload_config(new_cfg, FC, "2025-09-09")
    assert res["old_policy_hash"] != res["new_policy_hash"] == led.policy_hash
    stale = led.apply(pid)
    assert stale["code"] == "stale_policy_config" and stale["detail"]["approved_under"] == res["old_policy_hash"] and led.head() == 1
    assert led.approve(pid, "reviewer")["ok"] is False  # status is 'approved': not reopened by a reload; a fresh proposal is needed
    fresh = propose(led, {"meta": 81.0})
    assert led.approve(fresh, "reviewer")["ok"] and led.apply(fresh)["ok"]
    ev = led.audit("config_reloaded")[-1]["details"]
    assert ev["old_policy_hash"] == res["old_policy_hash"] and ev["new_policy_hash"] == res["new_policy_hash"]
    assert ev["old_experiment_config_hash"] == ev["new_experiment_config_hash"] == led.experiment_config_hash  # a policy reload keeps the experiment configuration


def test_a_proposal_built_under_a_different_configuration_is_refused(tmp_path):
    led = new_ledger(tmp_path)
    st = led.state()
    items = [ProposalItem(channel=c, action="hold", current_budget=b, new_budget=b, reason="r") for c, b in st["budgets"].items()]
    items[0] = ProposalItem(channel="meta", action="decrease", current_budget=90.0, new_budget=81.0, reason="r")
    p = Proposal(proposal_id="P-other-config", as_of="2025-09-08", ledger_version=1, policy_hash="a" * 64, items=items)
    led.create_proposal(p, "2025-09-08")
    r = led.approve("P-other-config", "reviewer")
    assert r["code"] == "stale_policy_config" and r["detail"]["proposal_built_under"] == "a" * 64 and r["detail"]["ledger_snapshot"] == led.policy_hash


def test_reopening_the_ledger_under_a_different_forecast_configuration_makes_approvals_stale(tmp_path):
    led = new_ledger(tmp_path)
    pid = propose(led, {"meta": 81.0})
    assert led.approve(pid, "reviewer")["ok"]
    led.close()
    other_fc = copy.deepcopy(FC)
    other_fc["forecast"]["interval_alpha"] = 0.10
    reopened = Ledger(tmp_path / "l.db", CFG, other_fc)
    r = reopened.apply(pid)
    assert r["code"] == "stale_policy_config" and reopened.head() == 1
    assert reopened.audit("apply_rejected")[-1]["details"]["code"] == "stale_policy_config"


# ---------------------------------------------------------------- audit


def test_audit_log_records_the_whole_story_and_is_append_only(tmp_path):
    led = new_ledger(tmp_path)
    pid, applied = approve_and_apply(led, {"meta": 81.0})
    led.apply(pid)  # duplicate attempt
    led.rollback(applied["change_id"], "reviewer", "undo", "2025-09-09")
    events = [e["event"] for e in led.audit()]
    assert events == ["ledger_initialized", "proposal_created", "approval_recorded", "apply_succeeded", "apply_rejected", "rollback_succeeded"]
    ok = led.audit("apply_succeeded")[0]
    assert ok["policy_hash"] == led.policy_hash and ok["proposal_id"] == pid and ok["ledger_version"] == 2
    d = ok["details"]
    assert d["approver"] == "reviewer" and d["total_before"] == 400.0 and d["total_after"] == 391.0 and d["reserve_before"] == 80.0 and d["reserve_after"] == 89.0
    assert led.audit("apply_rejected")[0]["details"]["code"] == "already_applied"
    for sql in ("UPDATE audit_log SET event = 'x'", "DELETE FROM audit_log", "UPDATE approvals SET approver = 'z'", "DELETE FROM ledger_versions",
                "UPDATE ledger_budgets SET budget = 1", "DELETE FROM proposals", "DELETE FROM applied_changes"):
        with pytest.raises(sqlite3.IntegrityError):
            led.conn.execute(sql)


def test_ledger_reopens_with_its_history(tmp_path):
    led = new_ledger(tmp_path)
    approve_and_apply(led, {"meta": 81.0})
    led.close()
    again = Ledger(tmp_path / "l.db", CFG, FC)
    assert again.head() == 2 and again.state()["budgets"]["meta"] == 81.0 and len(again.audit()) == 4
