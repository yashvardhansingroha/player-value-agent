"""SIMULATED experiment plan: canonical content, hash, validation, and binding to a ledger proposal item.
ALL DATA IS SYNTHETIC; every experiment here is a simulation.

A plan is a plain JSON-able dict. Its identity is the SHA-256 of its canonical JSON, so ANY change to ANY field (the assignment, a
unit budget, the estimator, a threshold, the keep/rollback rule, the power numbers) gives a different hash. The ledger stores the
full canonical plan, recomputes the hash and checks the plan against the proposal item and the ledger state before it lets an
increase through approval or application. Nothing in this module reads hidden truth or any outcome.
"""
import hashlib
import math
from datetime import date, timedelta
from pathlib import Path

import yaml

from src.ledger.proposal import canonical

ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_HORIZON_DAYS = 30  # the only outcome horizon the estimator, the simulator and the availability model support (D30)
SCHEMA = "experiment_plan_v1"
LABEL = "SIMULATED EXPERIMENT PLAN. ALL DATA IS SYNTHETIC."
REQUIRED = ("schema", "simulated", "label", "channel", "decision_as_of", "policy_hash", "experiment_config_hash", "budget_step", "target",
            "units", "assignment", "windows", "estimator", "guardrails", "decision_rule", "power_assessment", "status")


def plan_hash(plan: dict) -> str:
    """SHA-256 of the canonical JSON of the whole plan (floats as given, keys sorted, no whitespace)."""
    return hashlib.sha256(canonical(plan).encode("utf-8")).hexdigest()


def load_experiment_config(path=None) -> dict:
    with open(path or ROOT / "config" / "experiment.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict) -> str:
    """SHA-256 of the canonical JSON of an experiment configuration (the value a plan stores as `experiment_config_hash`)."""
    return hashlib.sha256(canonical(cfg).encode("utf-8")).hexdigest()


def standards(cfg: dict) -> dict:
    """The adequacy standards a plan must carry, taken from the AUTHORITATIVE experiment configuration."""
    pw = cfg["power"]
    return {"alternative_multiple_of_target": pw["alternative_multiple_of_target"], "target_power": pw["target_power"], "boundary_multiple": pw["boundary_multiple"],
            "below_boundary_multiples": pw["below_boundary_multiples"], "max_false_keep": pw["max_false_keep"],
            "false_keep_tolerance_mc_se": pw["false_keep_tolerance_mc_se"], "min_units_per_arm": cfg["design"]["min_units_per_arm"]}


def expected_estimator(cfg: dict) -> dict:
    return dict(cfg["estimator"], delta_method_se=True, critical_value="one-sided Student t, df = nT + nC - 2")


def micro(x) -> int:
    """Budgets are compared in millionths so 'exactly the approved step' is exact integer arithmetic, not float luck."""
    return int(round(float(x) * 1_000_000))


def add_days(iso: str, n: int) -> str:
    return (date.fromisoformat(iso) + timedelta(days=n)).isoformat()


def days_between(a: str, b: str) -> int:
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def adequacy_from_numbers(pa: dict) -> tuple:
    """Recompute 'is the design adequate?' from the NUMBERS recorded in the power assessment, so a plan cannot simply claim it.
    Rules (standards live inside the plan): at the chosen treatment window, power at the declared alternative >= target power, and
    false-keep at the boundary and below it <= max_false_keep + tolerance x Monte Carlo standard error."""
    std = pa["standards"]
    chosen = pa.get("chosen_treatment_days")
    if chosen is None:
        return False, ["no allowed treatment window met the adequacy standards"]
    block = pa["per_duration"].get(str(chosen))
    if block is None:
        return False, [f"no power numbers recorded for the chosen window {chosen}"]
    reasons = []
    by_mult = {round(a["multiple"], 6): a for a in block["alternatives"]}
    alt = by_mult.get(round(std["alternative_multiple_of_target"], 6))
    if alt is None or alt["keep_rate"] < std["target_power"]:
        reasons.append(f"power at {std['alternative_multiple_of_target']}x target is "
                       f"{'missing' if alt is None else round(alt['keep_rate'], 3)}, below {std['target_power']}")
    for m in [std["boundary_multiple"], *std["below_boundary_multiples"]]:
        a = by_mult.get(round(m, 6))
        if a is None:
            reasons.append(f"no false-keep estimate at {m}x target")
        elif a["keep_rate"] > std["max_false_keep"] + std["false_keep_tolerance_mc_se"] * a["mc_se"]:
            reasons.append(f"false-keep at {m}x target is {round(a['keep_rate'], 3)}, above {std['max_false_keep']} (+{std['false_keep_tolerance_mc_se']} MC SE)")
    if block.get("replicates", 0) < 200:
        reasons.append("too few replicates to judge")
    return (not reasons), reasons


def validate_plan_structure(plan: dict) -> list:
    """Internal consistency only (no ledger needed). Returns a list of error strings; empty means structurally valid."""
    err = []
    missing = [k for k in REQUIRED if k not in plan]
    if missing:
        return [f"missing fields: {missing}"]
    if plan["schema"] != SCHEMA or plan["simulated"] is not True:
        err.append("schema/simulated flag wrong")
    bs, units, w = plan["budget_step"], plan["units"], plan["windows"]
    if not (bs["proposed_daily_budget"] > bs["baseline_daily_budget"] > 0):
        err.append("budget step is not an increase of a positive baseline")
    if micro(bs["proposed_daily_budget"]) - micro(bs["baseline_daily_budget"]) != micro(bs["step_amount"]):
        err.append("step_amount is not proposed minus baseline")
    ids = [u["unit_id"] for u in units]
    if len(set(ids)) != len(ids):
        err.append("duplicate unit ids")
    treat = [u for u in units if u["arm"] == "treatment"]
    ctrl = [u for u in units if u["arm"] == "control"]
    if len(treat) + len(ctrl) != len(units):
        err.append("every unit must be 'treatment' or 'control'")
    mu = plan["power_assessment"]["standards"].get("min_units_per_arm", 0)
    if len(treat) < mu or len(ctrl) < mu:
        err.append(f"fewer than {mu} units in an arm ({len(treat)} treatment, {len(ctrl)} control)")
    # the units PARTITION the channel's baseline budget, and treated increments add up to EXACTLY the approved step
    if sum(micro(u["baseline_daily_budget"]) for u in units) != micro(bs["baseline_daily_budget"]):
        err.append("unit baseline budgets do not add up to the channel baseline budget")
    if sum(micro(u["planned_incremental_spend"]) for u in treat) != micro(bs["step_amount"]):
        err.append("treatment-unit incremental budgets do not add up to the exact channel-level step")
    if any(u["planned_incremental_spend"] != 0 for u in ctrl) or any(u["planned_incremental_spend"] <= 0 for u in treat):
        err.append("control units must have zero increment and treatment units a positive one")
    # windows
    lag, horizon = w["reporting_lag_days"], w["outcome_horizon_days"]
    if days_between(w["pre_start"], w["pre_end"]) + 1 != w["pre_days"] or days_between(w["treatment_start"], w["treatment_end"]) + 1 != w["treatment_days"]:
        err.append("window lengths do not match their dates")
    if w["treatment_start"] != plan["decision_as_of"]:
        err.append("treatment must start on the decision date")
    if add_days(w["pre_end"], horizon + lag) > plan["decision_as_of"]:
        err.append("pre-period outcomes would not be available at the decision date")
    if w["readout_available_on"] != add_days(w["treatment_end"], horizon + lag):
        err.append("readout_available_on is not last acquisition day + D30 horizon + reporting lag")
    if plan["power_assessment"].get("chosen_treatment_days") not in (None, w["treatment_days"]):
        err.append("treatment window differs from the window the power study chose")
    # the decision rule must name the estimator and a numeric target
    if not isinstance(plan["decision_rule"].get("target_d30_roas"), (int, float)) or plan["decision_rule"]["target_d30_roas"] != plan["target"]["target_d30_roas"]:
        err.append("decision-rule target differs from the plan target")
    if plan["estimator"].get("name") != "unit_level_did_ratio":
        err.append("unsupported estimator")
    ok, reasons = adequacy_from_numbers(plan["power_assessment"])
    if ok != bool(plan["power_assessment"].get("adequate")):
        err.append(f"recorded adequacy ({plan['power_assessment'].get('adequate')}) disagrees with the recorded power numbers ({ok})")
    return err


def design_fingerprint(plan: dict) -> str:
    """Hash of the DESIGN the planning study was run on: channel, date, exact step, target, units/arms/increments, assignment, estimator,
    guardrails and the pre-period. The study records it next to its numbers, so the recorded numbers cannot be silently re-attached to a
    different design (another step, target, assignment or threshold). It does NOT prove the numbers were really produced by the study."""
    w = plan["windows"]
    core = {"channel": plan["channel"], "decision_as_of": plan["decision_as_of"], "budget_step": plan["budget_step"], "target_d30_roas": plan["target"]["target_d30_roas"],
            "units": plan["units"], "assignment": plan["assignment"], "estimator": plan["estimator"], "guardrails": plan["guardrails"],
            "pre": [w["pre_start"], w["pre_end"], w["pre_days"], w["reporting_lag_days"], w["outcome_horizon_days"]]}
    return hashlib.sha256(canonical(core).encode("utf-8")).hexdigest()


def _chosen_from_numbers(pa: dict, options: list):
    """Shortest allowed treatment window whose RECORDED numbers meet the plan's standards, or None."""
    for n in options:
        ok, _ = adequacy_from_numbers({"standards": pa["standards"], "chosen_treatment_days": n, "per_duration": pa["per_duration"]})
        if ok:
            return n
    return None


def validate_plan_against_config(plan: dict, cfg: dict) -> list:
    """The plan must follow the AUTHORITATIVE experiment configuration `cfg` (held by the ledger, never taken from the plan). Returns error
    strings; empty means the plan follows it. Checks: the config hashes to the plan's `experiment_config_hash`; the adequacy standards,
    estimator settings, alpha, guardrails, allowed duration, pre-period, horizon and unit counts are the configuration's; the recorded
    planning results have the declared replicates/seed/alternatives, internally consistent Monte Carlo errors, the design fingerprint of
    THIS plan, and the chosen duration the standards would pick.

    TRUST BOUNDARY (documented, not removed): this does not re-run the planning study. A party who can write a plan can still invent
    keep rates that satisfy the standards for a design it did not simulate. What cannot be done any more is to weaken the standards,
    the estimator, the guardrails, the durations or the thresholds, to attach numbers to a different design, or to claim inconsistent
    Monte Carlo errors. Re-running the study (it is deterministic and cheap) is the independent check."""
    try:
        return _validate_against_config(plan, cfg)
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as e:
        return [f"plan content is malformed for the configuration check: {type(e).__name__}: {e}"]


def _validate_against_config(plan: dict, cfg: dict) -> list:
    err = []
    if config_hash(cfg) != plan["experiment_config_hash"]:
        return ["the authoritative experiment configuration does not hash to the plan's experiment_config_hash"]
    de, pw, pa, w = cfg["design"], cfg["power"], plan["power_assessment"], plan["windows"]
    if pa["standards"] != standards(cfg):
        err.append(f"adequacy standards differ from the configuration's (plan {pa['standards']}, configuration {standards(cfg)})")
    if plan["estimator"] != expected_estimator(cfg):
        err.append("estimator settings differ from the configuration's")
    if plan["guardrails"] != cfg["guardrails"]:
        err.append("guardrail settings differ from the configuration's")
    if plan["decision_rule"].get("alpha") != cfg["estimator"]["alpha"]:
        err.append("decision-rule alpha differs from the configuration's estimator alpha")
    options = list(de["treatment_day_options"])
    if w["treatment_days"] not in options:
        err.append(f"treatment window {w['treatment_days']} is not an allowed duration {options}")
    if w["pre_days"] != de["pre_days"]:
        err.append(f"pre-period length {w['pre_days']} differs from the configuration's {de['pre_days']}")
    if w["outcome_horizon_days"] != SUPPORTED_HORIZON_DAYS:
        err.append(f"outcome horizon {w['outcome_horizon_days']} is not the supported D{SUPPORTED_HORIZON_DAYS}")
    if len(plan["units"]) != de["units_per_channel"]:
        err.append(f"{len(plan['units'])} units, configuration requires {de['units_per_channel']}")
    # recorded planning results
    if pa.get("replicates") != pw["replicates"] or pa.get("seed") != pw["seed"]:
        err.append("recorded planning replicates/seed differ from the configuration's")
    if sorted(pa["per_duration"]) != sorted(str(n) for n in options):
        err.append(f"planning numbers are recorded for windows {sorted(pa['per_duration'])}, not for all allowed durations {options}")
    target = plan["target"]["target_d30_roas"]
    for key, blk in pa["per_duration"].items():
        n = blk["replicates"]
        if blk["treatment_days"] != int(key) or n != pw["replicates"]:
            err.append(f"planning block {key} has the wrong duration or replicate count")
        # every configured multiple must be present; the study may add informational extras (they take no part in adequacy, and each is still checked below)
        if not {round(m, 6) for m in pw["reported_multiples"]} <= {round(a["multiple"], 6) for a in blk["alternatives"]}:
            err.append(f"planning block {key} does not report all the configured multiples")
        for a in blk["alternatives"]:
            p = a["keep_rate"]
            if not (0.0 <= p <= 1.0) or abs(p * n - round(p * n)) > 1e-6:
                err.append(f"block {key} multiple {a['multiple']}: keep rate {p} is not a count out of {n} replicates")
            elif abs(a["mc_se"] - math.sqrt(max(p * (1 - p), 1e-12) / n)) > 1e-9:
                err.append(f"block {key} multiple {a['multiple']}: Monte Carlo SE {a['mc_se']} is not binomial sqrt(p(1-p)/n)")
            if abs(a["true_iroas"] - a["multiple"] * target) > 1e-9 * max(1.0, abs(target)):
                err.append(f"block {key} multiple {a['multiple']}: declared true iROAS is not multiple x this plan's target")
    if _chosen_from_numbers(pa, options) != pa.get("chosen_treatment_days"):
        err.append(f"chosen treatment window {pa.get('chosen_treatment_days')} is not the shortest window the standards accept "
                   f"({_chosen_from_numbers(pa, options)}) given the recorded numbers")
    if pa.get("design_fingerprint") != design_fingerprint(plan):
        err.append("the planning numbers were not recorded for THIS design (design fingerprint missing or different)")
    return err


def validate_plan_binding(plan: dict, item: dict, state: dict, policy_hash_value: str, proposal_as_of: str) -> list:
    """Is THIS plan the plan for THIS proposal item on THIS ledger state? Returns error strings (empty = bound)."""
    err = []
    bs = plan["budget_step"]
    if plan["channel"] != item["channel"]:
        err.append(f"plan channel {plan['channel']!r} differs from the item channel {item['channel']!r}")
    if micro(bs["baseline_daily_budget"]) != micro(item["current_budget"]) or micro(bs["baseline_daily_budget"]) != micro(state["budgets"].get(item["channel"], -1)):
        err.append("plan baseline budget differs from the item's current budget or the ledger's current budget")
    if micro(bs["proposed_daily_budget"]) != micro(item["new_budget"]):
        err.append("plan proposed budget differs from the item's new budget (the exact step)")
    if plan["policy_hash"] != policy_hash_value:
        err.append("plan was frozen under a different policy configuration")
    if plan["decision_as_of"] != proposal_as_of:
        err.append("plan decision date differs from the proposal's as_of")
    ev = (item.get("evidence") or {}).get("payback_target")
    if ev and ev.get("target_d30_roas") is not None and ev["target_d30_roas"] != plan["target"]["target_d30_roas"]:
        err.append("plan target differs from the target in the proposal's evidence")
    return err
