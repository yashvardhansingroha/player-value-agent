"""SIMULATED experiment readout. ALL DATA IS SYNTHETIC. Decision-facing: sees ONLY the frozen plan and the observed unit-day rows.

It never imports the simulator or hidden truth. It refuses to run
  * before the last acquisition cohort's D30 outcome and reporting delay have elapsed on the simulation clock (ReadoutNotAvailable),
  * if any expected observation row is missing (ReadoutIncomplete),
  * under a policy / forecast / experiment configuration different from the one the plan was frozen under (ConfigurationMismatch).
The estimator and the keep/rollback rule are exactly the functions the planning simulation used (design.estimate / design.decide),
with every threshold read from the FROZEN PLAN, not from any config.
"""
import numpy as np
import pandas as pd

from src.experiments import design as D
from src.experiments import plan as P
from src.policy.constraints import policy_hash


class ReadoutNotAvailable(Exception):
    def __init__(self, available_on, clock, days_remaining, pending_days):
        super().__init__(f"readout not available: needs the simulation clock to reach {available_on} (clock is {clock}, {days_remaining} day(s) to go)")
        self.available_on, self.clock, self.days_remaining, self.pending_days = available_on, clock, days_remaining, pending_days


class ReadoutIncomplete(Exception):
    pass


class ConfigurationMismatch(Exception):
    pass


def readout_status(plan: dict, clock: str, obs: pd.DataFrame) -> dict:
    """Is the experiment still pending? Which acquisition days still lack an available D30 outcome?"""
    w = plan["windows"]
    have = set(obs[obs.phase == "treatment"].acquisition_date)
    days = pd.date_range(w["treatment_start"], w["treatment_end"]).strftime("%Y-%m-%d").tolist()
    pending = [d for d in days if d not in have]
    remaining = max(0, P.days_between(clock, w["readout_available_on"]))
    return {"state": "pending" if clock < w["readout_available_on"] else "available", "available_on": w["readout_available_on"], "clock": clock,
            "days_remaining": remaining, "pending_acquisition_days": pending}


def run_readout(plan: dict, obs: pd.DataFrame, clock: str, policy_cfg: dict, forecast_cfg: dict, experiment_cfg: dict) -> dict:
    w = plan["windows"]
    status = readout_status(plan, clock, obs)
    if clock < w["readout_available_on"]:
        raise ReadoutNotAvailable(w["readout_available_on"], clock, status["days_remaining"], status["pending_acquisition_days"])
    loaded = {"policy_hash": policy_hash(policy_cfg, forecast_cfg), "experiment_config_hash": D.config_hash(experiment_cfg)}
    if loaded["policy_hash"] != plan["policy_hash"] or loaded["experiment_config_hash"] != plan["experiment_config_hash"]:
        raise ConfigurationMismatch(f"the loaded configuration {loaded} differs from the one the plan was frozen under "
                                    f"({{'policy_hash': {plan['policy_hash']!r}, 'experiment_config_hash': {plan['experiment_config_hash']!r}}}); load the plan's configuration explicitly")
    h = P.plan_hash(plan)
    avail = obs[obs.available_on <= clock]
    if "plan_hash" in avail:  # a treatment row counts only if it was generated for THIS plan; pre-period history belongs to no plan
        avail = avail[(avail.phase == "pre") | (avail.plan_hash == h)]
    try:
        st, unit_order, problems = D.aggregate_observations(avail, plan)
    except ValueError as e:
        raise ReadoutIncomplete(str(e)) from e
    rule = D.rule_from_plan(plan)
    treat = D.treat_mask(plan, unit_order)
    planned_mean = D.planned_mean_increment(plan)
    est = D.estimate(st, treat, planned_mean, rule["alpha"])
    code = int(D.decide(est, rule)[0])
    f = lambda k: _finite_or_none(est[k][0])  # a missing or infinite statistic is reported as None (unavailable), never as a number
    n_t, days = int(est["n_treatment"]), w["treatment_days"]
    planned_total_per_day = float(sum(u["planned_incremental_spend"] for u in plan["units"] if u["arm"] == "treatment"))
    extra_spend = f("extra_spend_per_day")
    realised_total_per_day = None if extra_spend is None else extra_spend * n_t  # mean treated-minus-control change x number of treated units
    lower = f("lower")
    return {
        "label": "SIMULATED READOUT. ALL DATA IS SYNTHETIC.", "plan_hash": h, "clock": clock, "loaded_configuration": loaded,
        "estimate": {"iroas": f("iroas"), "se": f("se"), "lower_bound": lower, "alpha": rule["alpha"], "df": int(est["df"]),
                     "extra_revenue_per_day": f("extra_revenue_per_day"), "extra_spend_per_day": extra_spend},
        "spending": {"planned_extra_spend_per_day_total": planned_total_per_day, "planned_extra_spend_over_window": planned_total_per_day * days,
                     "realised_extra_spend_per_day_total_estimate": realised_total_per_day,
                     "realised_extra_spend_over_window_estimate": None if realised_total_per_day is None else realised_total_per_day * days,
                     "delivery_fraction": f("delivery_fraction"), "delivery_bounds": [rule["min_delivery_fraction"], rule["max_delivery_fraction"]]},
        "guardrails": {"cpi_ratio": f("cpi_ratio"), "cpi_ratio_limit": rule["max_cpi_ratio"], "d1_retention_change": f("d1_retention_change"),
                       "d1_retention_change_floor": rule["min_d1_retention_change"]},
        "rule": {"target_d30_roas": rule["target_d30_roas"], "lower_bound_meets_target": bool(lower is not None and lower >= rule["target_d30_roas"])},
        "units": {"treatment": n_t, "control": int(est["n_control"])},
        "input_problems": problems, "invalid_readout": code == 4,
        "decision": "keep" if code == 0 else "rollback", "decision_code": D.CODES[code],
    }


def _finite_or_none(x):
    x = float(x)
    return x if np.isfinite(x) else None
