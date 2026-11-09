"""HIDDEN-SIDE scoring of a simulated experiment AFTER the readout. SIMULATED. ALL DATA IS SYNTHETIC.

Allowed to use the simulator's truth because it runs only after the decision and never feeds one. It answers: how close was the
estimate to the simulated truth, did the one-sided lower bound sit below the truth, was keep/rollback the right call (truth >= target),
and, by re-running the actual simulator many times on fresh seeds with the SAME estimator and rule, how often would this design keep
at the true value (evaluation-side power and false-keep, to be compared with the planning study)."""
import numpy as np

from src.experiments import design as D


def score_readout(plan: dict, readout: dict, sim) -> dict:
    truth = sim.true_effect(plan)
    target = plan["target"]["target_d30_roas"]
    true = truth["true_iroas_treated_units"]
    est = readout["estimate"]
    valid = est["iroas"] is not None and est["lower_bound"] is not None  # an invalid readout has no estimate to score
    return {"label": "SIMULATED: scored against the simulator's truth after the decision",
            "true_iroas_treated_units": true, "estimated_iroas": est["iroas"], "estimation_error": est["iroas"] - true if valid else None,
            "lower_bound": est["lower_bound"], "lower_bound_at_or_below_truth": bool(est["lower_bound"] <= true) if valid else None,
            "target_d30_roas": target, "truth_meets_target": bool(true >= target),
            "decision": readout["decision"], "decision_correct": bool((readout["decision"] == "keep") == (true >= target)),
            # channel TOTALS per day over all treated units (the readout's N and D are per-unit means, so they are multiplied by the treated count)
            "true_extra_spend_per_day_total": truth["true_extra_spend_per_day"],
            "estimated_realised_extra_spend_per_day_total": readout["spending"]["realised_extra_spend_per_day_total_estimate"],
            "planned_extra_spend_per_day_total": readout["spending"]["planned_extra_spend_per_day_total"],
            "true_extra_revenue_per_day_total": truth["true_extra_revenue_per_day"],
            "estimated_extra_revenue_per_day_total": None if est["extra_revenue_per_day"] is None else est["extra_revenue_per_day"] * readout["units"]["treatment"],
            "reference_channel_level_proportional_step_roas": truth["reference_channel_level_proportional_step_roas"]}


def replicate_study(plan: dict, sim, n: int, seed: int, reference_estimate: float = None) -> dict:
    """Re-run the ACTUAL simulator n times (fresh noise) through the SAME estimator and rule used by the readout.
    The assignment and unit parameters are held fixed, so the spread of estimates reflects observation noise only (the readout's
    delta-method SE also reflects unit-to-unit heterogeneity, as if assignment were random, and is therefore larger).
    `reference_estimate` (for example the actual readout's iROAS) is located within the replicate distribution."""
    rule = D.rule_from_plan(plan)
    treat = D.treat_mask(plan, sim.unit_ids)
    mean_inc = D.planned_mean_increment(plan)
    truth = sim.true_effect(plan)["true_iroas_treated_units"]
    iroas, lower, codes, se = [], [], [], []
    for i in range(n):
        a = sim.arrays(plan, D.derive_seed(seed, "replicate", i))
        st = D.unit_stats(a["spend"], a["installs"], a["d1_active"], a["d30_net_revenue"], plan["windows"]["pre_days"])
        est = D.estimate(st, treat, mean_inc, rule["alpha"])
        iroas.append(float(est["iroas"][0]))
        lower.append(float(est["lower"][0]))
        se.append(float(est["se"][0]))
        codes.append(int(D.decide(est, rule)[0]))
    iroas, lower, codes, se = np.array(iroas), np.array(lower), np.array(codes), np.array(se)
    keep = float((codes == 0).mean())
    return {"label": "SIMULATED evaluation-side study (uses truth)", "replicates": n, "true_iroas_treated_units": truth, "target_d30_roas": rule["target_d30_roas"],
            "keep_rate": keep, "keep_rate_mc_se": float(np.sqrt(max(keep * (1 - keep), 1e-12) / n)),
            "estimate_mean": float(np.nanmean(iroas)), "estimate_bias": float(np.nanmean(iroas) - truth), "estimate_sd": float(np.nanstd(iroas)),
            "lower_bound_coverage_of_truth": float((lower <= truth).mean()), "nominal_one_sided_coverage": 1 - rule["alpha"],
            "median_delta_method_se": float(np.nanmedian(se)), "estimate_quantiles_5_50_95": [float(np.nanquantile(iroas, q)) for q in (0.05, 0.5, 0.95)],
            "reference_estimate": reference_estimate,
            "reference_estimate_percentile": None if reference_estimate is None else float((iroas < reference_estimate).mean()),
            "rollback_reasons": {D.CODES[c]: float((codes == c).mean()) for c in (1, 2, 3, 4)}}
