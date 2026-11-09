"""SIMULATED experiment lifecycle glue between the experiment store and the sandbox ledger. ALL DATA IS SYNTHETIC.
plan -> approval -> application -> pending -> readout -> keep or rollback. Decision-facing: it never imports the simulator.

Rules enforced here (each tested):
  * the experiment proposal contains ONLY the experiment-backed increase (every other channel is a hold), so rolling the experiment back
    can never undo an unrelated change that was approved together with it;
  * `execute_readout` never trusts a caller-supplied decision: it re-verifies the plan and the run link, reads the store's own clock and
    observations, recomputes the readout with the production function under explicitly supplied configuration, and executes only that;
  * a readout decision is executed at most once per plan (UNIQUE outcome); a `rollback_blocked` result records NO outcome, so it can be
    retried after the conflict is resolved, and is never silently treated as a keep;
  * the ledger must already hold the configuration the plan was frozen under; this module never reloads it for you.
"""
import json

from src.experiments import plan as P
from src.experiments import readout as R
from src.experiments import store as E
from src.ledger.ledger import Reject
from src.ledger.proposal import Proposal, ProposalItem, jsonable, make_proposal_id


def decision_after_design(decision: dict, plan: dict) -> dict:
    """If the experiment design is inadequate the increase test cannot be run: the channel's action becomes insufficient_evidence and its
    budget stays where it was. An adequate plan leaves the policy decision as it was (the increase still needs approval and application)."""
    out = dict(decision)
    pa = plan["power_assessment"]
    if pa["adequate"]:
        out["experiment_design"] = {"adequate": True, "chosen_treatment_days": pa["chosen_treatment_days"], "plan_hash": P.plan_hash(plan)}
        return out
    out.update(action="insufficient_evidence", proposed_budget=decision["current_budget"],
               reasons=["experiment design is inadequate, so the increase cannot be tested: " + "; ".join(pa["reasons"])] + list(decision["reasons"]),
               increase_candidate=None)
    out["experiment_design"] = {"adequate": False, "reasons": pa["reasons"], "plan_hash": P.plan_hash(plan)}
    return out


def experiment_proposal(source: Proposal, channel: str, plan: dict, ledger_version: int) -> Proposal:
    """Copy of the policy proposal in which only `channel` carries a change: an experiment-backed increase bound to `plan`."""
    h = P.plan_hash(plan)
    items = []
    for it in source.items:
        if it.channel == channel:
            if it.action != "increase_test":
                raise ValueError(f"{channel} is not an increase test in the source proposal")
            items.append(ProposalItem(channel=channel, action="increase_test", current_budget=it.current_budget, new_budget=it.new_budget,
                                      qualification="experiment_backed", experiment_plan_hash=h, evidence=it.evidence,
                                      reason=it.reason + " | bound to a frozen SIMULATED experiment plan"))
        else:
            items.append(ProposalItem(channel=it.channel, action="hold", current_budget=it.current_budget, new_budget=it.current_budget,
                                      qualification="none", reason="not part of this experiment proposal"))
    return Proposal(proposal_id=make_proposal_id(source.as_of, items, ledger_version, source.policy_hash) + "-EXP", as_of=source.as_of,
                    ledger_version=ledger_version, policy_hash=source.policy_hash, source=source.source, items=items)


def advance_clock(ec, sim, plan: dict, new_clock: str, seed_root: int) -> dict:
    """Move the simulation clock forward and store every treatment observation that has become AVAILABLE by then. `sim` is any object
    with `treatment_rows_up_to` (the hidden-side simulator, passed in by the caller: this module does not import it). The plan must
    already be frozen: the database refuses treatment outcomes otherwise. Outcomes not yet available are simply not generated."""
    h = P.plan_hash(plan)
    E.set_clock(ec, new_clock)
    done = {r[0] for r in ec.execute("SELECT DISTINCT acquisition_date FROM observations WHERE plan_hash = ?", (h,))}
    rows = sim.treatment_rows_up_to(plan, h, new_clock, seed_root, done)
    if rows:
        E.insert_observations(ec, rows)
    E.log_event(ec, "observations_released", h, {"clock": new_clock, "new_rows": len(rows)})
    return {"clock": new_clock, "new_rows": len(rows), "total_treatment_rows": E.treatment_rows(ec, h)}


def start_experiment(ledger, ec, plan_hash: str, change_id: str) -> dict:
    """Link a registered plan to the ledger change that applied it. Both sides are VERIFIED: the plan must be registered in the ledger
    (and re-verify now) and frozen in the experiment store with identical content, and `change_id` must be the active experiment-backed
    APPLY of exactly this plan's step. The link is dated by the experiment store's clock. Raises ValueError otherwise."""
    try:
        plan = ledger.verified_plan(plan_hash)
    except Reject as r:
        raise ValueError(f"plan {plan_hash} cannot be started: {r.code} {r.detail}") from r
    if E.get_plan(ec, plan_hash) != plan:
        raise ValueError(f"plan {plan_hash} is not frozen in the experiment store with the content the ledger holds")
    errors = ledger.verify_experiment_change(change_id, plan_hash)
    if errors:
        raise ValueError(f"change {change_id} is not the applied change for plan {plan_hash}: {errors}")
    applied_on = E.clock(ec)
    ec.execute("INSERT INTO experiment_runs VALUES (?,?,?)", (plan_hash, change_id, applied_on))
    E.log_event(ec, "experiment_started", plan_hash, {"change_id": change_id, "applied_on": applied_on, "readout_available_on": plan["windows"]["readout_available_on"]})
    return {"plan_hash": plan_hash, "change_id": change_id, "applied_on": applied_on}


def _comparable(readout: dict) -> dict:
    """A readout without the clock field, as plain JSON (so two readouts compare by value)."""
    return json.loads(json.dumps(jsonable({k: v for k, v in readout.items() if k != "clock"}), sort_keys=True))


def execute_readout(ledger, ec, plan_hash: str, requested_by: str, *, policy_cfg: dict, forecast_cfg: dict, experiment_cfg: dict, preview: dict = None) -> dict:
    """Carry out the keep/rollback decision of the experiment `plan_hash` through the ledger. AUTHORITATIVE: nothing the caller says about
    the result is trusted. The decision executed is the one RECOMPUTED here, by the production readout function, from
      * the plan registered in the ledger (re-verified now: not altered, valid, follows the ledger's experiment configuration, adequate),
        which must be identical to the plan frozen in the experiment store;
      * the run link, which must point at the active experiment-backed ledger change for exactly this plan;
      * the experiment store's own clock and the observations it holds (a pending or incomplete experiment cannot be read out);
      * the explicitly supplied policy / forecast / experiment configurations, which must be the ones the plan and the ledger hold.
    A caller-supplied `preview` readout is only compared with the recomputed one; any difference rejects. Every refusal returns
    {"status": ..., "code": ..., "outcome_recorded": False}, records NO completed outcome and changes NO budget. Returns a status dict
    and never raises for a refusal."""
    def refuse(code, **details):
        E.log_event(ec, "readout_execution_rejected", plan_hash, {"code": code, **jsonable(details)})
        return {"status": "rejected", "code": code, "outcome_recorded": False, **jsonable(details)}

    clock = E.clock(ec)  # the experiment store's simulation clock is the only time that matters here
    try:
        plan = ledger.verified_plan(plan_hash)
    except Reject as r:
        return refuse(r.code, **r.detail)
    stored = E.get_plan(ec, plan_hash)
    if stored is None or P.plan_hash(stored) != plan_hash or stored != plan:
        return refuse("plan_not_frozen_in_experiment_store", detail="the experiment store does not hold this exact plan")
    run = ec.execute("SELECT change_id FROM experiment_runs WHERE plan_hash = ?", (plan_hash,)).fetchone()
    if run is None:
        return refuse("experiment_not_applied")
    change_id = run[0]
    if ec.execute("SELECT 1 FROM experiment_outcomes WHERE plan_hash = ?", (plan_hash,)).fetchone():
        return refuse("readout_already_executed")
    errors = ledger.verify_experiment_change(change_id, plan_hash)
    if errors:
        return refuse("experiment_link_invalid", linked_change_id=change_id, errors=errors)
    if ledger.policy_hash != plan["policy_hash"]:
        details = {"ledger_policy_hash": ledger.policy_hash, "plan_policy_hash": plan["policy_hash"],
                   "action": "load the configuration the plan was frozen under into the ledger with reload_config before executing the readout"}
        E.log_event(ec, "readout_execution_rejected", plan_hash, {"code": "configuration_not_loaded", **details})
        ledger.record_event("experiment_readout_rejected", clock, change_id, {"code": "configuration_not_loaded", **details})
        return {"status": "configuration_not_loaded", "outcome_recorded": False, **details}
    try:
        readout = R.run_readout(plan, E.load_observations(ec), clock, policy_cfg, forecast_cfg, experiment_cfg)
    except R.ReadoutNotAvailable as e:
        return refuse("readout_not_available", available_on=e.available_on, clock=e.clock, days_remaining=e.days_remaining, pending_days=len(e.pending_days))
    except R.ReadoutIncomplete as e:
        return refuse("readout_incomplete", detail=str(e))
    except R.ConfigurationMismatch as e:
        return refuse("configuration_mismatch", detail=str(e))
    if preview is not None:
        mine, theirs = _comparable(readout), _comparable(preview)
        if mine != theirs:
            return refuse("readout_preview_mismatch", differing_fields=sorted(k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k)),
                          detail="the supplied readout is only a preview and differs from the recomputed one; the recomputed readout was NOT executed")
    if readout["decision"] == "keep":
        ledger.record_event("experiment_kept", clock, change_id, {"plan_hash": plan_hash, "estimate": readout["estimate"], "rule": readout["rule"]})
        res = {"status": "kept", "change_id": change_id}
        outcome = "keep"
    else:
        rb = ledger.rollback(change_id, requested_by, f"experiment readout: {readout['decision_code']}", clock)
        if rb["status"] != "rolled_back":
            E.log_event(ec, "rollback_not_executed", plan_hash, {"ledger_status": rb["status"], "code": rb.get("code")})
            ledger.record_event("experiment_rollback_not_executed", clock, change_id, {"plan_hash": plan_hash, "ledger_status": rb["status"], "code": rb.get("code")})
            return {"status": "rollback_blocked" if rb["status"] == "rollback_blocked" else "rollback_rejected", "ledger": jsonable(rb), "outcome_recorded": False,
                    "readout": readout, "note": "the readout says ROLL BACK but the ledger could not; no outcome was recorded and the increase is NOT treated as kept"}
        res = {"status": "rolled_back", "change_id": change_id, "rollback_change_id": rb["change_id"], "before": rb["before"], "after": rb["after"]}
        outcome = "rolled_back"
    res["readout"] = readout
    ec.execute("INSERT INTO experiment_outcomes VALUES (?,?,?,?,?)", (plan_hash, outcome, json.dumps(jsonable(readout), sort_keys=True),
                                                                     json.dumps(jsonable({k: v for k, v in res.items() if k != "readout"}), sort_keys=True), clock))
    E.log_event(ec, "readout_executed", plan_hash, {"outcome": outcome})
    return res
