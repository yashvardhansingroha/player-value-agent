"""Test helpers for the SIMULATED experiment machinery. `manual_plan` builds a structurally valid plan that follows the experiment
configuration, with FABRICATED power numbers (so ledger tests are fast); it is never used as evidence about statistical performance,
which comes from the real planning study. The fabricated numbers still have the shape the real study records: every allowed duration,
the configured multiples, binomial Monte Carlo errors, alternatives scaled by THIS plan's target, and the design fingerprint."""
import math

from src.experiments import design as D
from src.experiments import plan as P

CFG = D.load_config()


def synthetic_truth(channel="meta", b=0.7, gamma=0.0, r1=0.4):
    """A tiny hidden-truth dict with the fields the simulator reads (power-law response, no quality dilution by default)."""
    return {"channels": {channel: {"response": {"type": "power", "a": 80 / 80 ** b, "b": b}, "quality_gamma": gamma, "base_spend": 80.0}},
            "resolved_config": {"start_date": "2025-01-06", "channels": {channel: {"retention": {"r1": r1, "alpha": 0.5}}}},
            "planted": {"regime_shift": {"start_day": 10_000, "iap_purchase_probability_multiplier": 0.7, "ecpm_multiplier": 1.1}}}


def _block(days: int, target: float, strong: bool, replicates: int) -> dict:
    keep = {0.5: 0.00, 0.8: 0.03, 1.0: 0.08, 1.25: 0.40, 1.5: 0.90 if strong else 0.30, 2.0: 0.99}
    alts = [{"multiple": m, "true_iroas": m * target, "keep_rate": p, "mc_se": math.sqrt(max(p * (1 - p), 1e-12) / replicates), "median_estimate": m * target,
             "median_lower_bound": 0.0, "median_se": 0.1, "rollback_reasons": {}} for m, p in keep.items()]
    return {"treatment_days": days, "replicates": replicates, "alternatives": alts}


def power_numbers(adequate: bool, treatment_days: int = 14, target: float = 0.5) -> dict:
    """Fabricated planning numbers: windows shorter than `treatment_days` lack power, `treatment_days` and longer pass (all fail when not adequate)."""
    reps = CFG["power"]["replicates"]
    per = {str(n): _block(n, target, adequate and n >= treatment_days, reps) for n in CFG["design"]["treatment_day_options"]}
    pa = {"standards": D.standards(CFG), "replicates": reps, "seed": CFG["power"]["seed"], "per_duration": per,
          "chosen_treatment_days": treatment_days if adequate else None, "note": "FABRICATED numbers for a unit test"}
    pa["adequate"], pa["reasons"] = P.adequacy_from_numbers(pa)
    return pa


def manual_plan(channel="meta", baseline=90.0, proposed=99.0, policy_hash="a" * 64, as_of="2025-09-08", adequate=True, k=24, treatment_days=14,
                lag=2, target=0.5, units=None, assign_seed=7, experiment_config_hash=None) -> dict:
    units = units or D.make_units(channel, baseline, k, 1)
    assignment = D.assign(units, assign_seed)
    step = round(proposed - baseline, 6)
    plan = {
        "schema": P.SCHEMA, "simulated": True, "label": P.LABEL, "channel": channel, "decision_as_of": as_of, "policy_hash": policy_hash,
        "experiment_config_hash": experiment_config_hash or D.config_hash(CFG),
        "budget_step": {"baseline_daily_budget": baseline, "proposed_daily_budget": proposed, "step_amount": step, "step_pct": round(step / baseline, 6)},
        "target": {"target_d30_roas": target, "source": {"note": "test"}}, "model_estimate": {"finite_step_marginal_roas": 0.6, "lower_bound": 0.5},
        "units": D.allocate_step(units, assignment["treated"], step), "assignment": assignment,
        "windows": D.windows(as_of, lag, CFG["design"]["pre_days"], treatment_days),
        "estimator": P.expected_estimator(CFG), "guardrails": dict(CFG["guardrails"]),
        "decision_rule": {"keep_if": "see plan", "target_d30_roas": target, "alpha": CFG["estimator"]["alpha"]},
        "power_assessment": power_numbers(adequate, treatment_days, target), "status": "frozen_before_outcomes",
    }
    plan["power_assessment"]["design_fingerprint"] = P.design_fingerprint(plan)
    return plan


def refingerprint(plan: dict) -> dict:
    """After a test edits a plan's DESIGN (units, thresholds, ...), re-attach a matching fingerprint so only the intended defect remains."""
    plan["power_assessment"]["design_fingerprint"] = P.design_fingerprint(plan)
    return plan
