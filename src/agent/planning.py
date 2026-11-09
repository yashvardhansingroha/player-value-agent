"""The REAL recalculation hooks for an agent context: they call the existing verified functions and add no number of their own.
ALL DATA IS SYNTHETIC; experiments are SIMULATED.

  step recalculation  policy.estimate_step on the decision's own response-curve evidence (the same function the policy used)
  experiment planner  design.build_plan: units, stratified assignment, exact step allocation, planning power study, adequacy standards

`pre_obs_provider(channel, units, windows)` supplies the experiment units' observed PRE-period history. In this project that history comes
from the SIMULATOR, which only the harness (scripts/) may import, so it is injected here; this module never touches hidden truth.
"""
import pandas as pd

from src.agent.context import AgentContext, NotRecalculable, r6
from src.data.metrics import milestone_complete_at
from src.experiments import design as D
from src.policy import policy as pol


def declared_reporting_lag(dec, as_of: str, channel: str) -> int:
    """The channel's declared maximum reporting lag in days: how long after day 3 its D3 milestone completes."""
    c = milestone_complete_at(dec, as_of, 3)
    c = c[c.channel == channel]
    return int(max((pd.Timestamp(a) - pd.Timestamp(b)).days - 3 for a, b in zip(c.complete_at, c.install_date)))


def make_step_recalc(dec_store, result: dict, as_of: str, pcfg: dict):
    decisions = result["decisions"]

    def step_recalc(channel: str, new_budget: float) -> dict:
        d = decisions[channel]
        ev = d["evidence"]
        resp, pol_step = ev.get("response_curve"), ev.get("marginal_d30_roas_step")
        if resp is None or pol_step is None or not resp.get("usable"):
            raise NotRecalculable("the policy did not evaluate an increase for this channel (no usable response curve in its evidence)")
        s0 = d["current_budget"]
        lo, hi = resp["spend_range"]
        if new_budget <= s0 or new_budget > hi + 1e-9:
            raise NotRecalculable(f"{new_budget} is not an increase inside the observed spend range [{lo:.2f}, {hi:.2f}]")
        if d["action"] == "increase_test" and abs(new_budget - d["proposed_budget"]) < 1e-6:
            est = pol_step  # the policy's own estimate for exactly this step
        else:
            S = pol.load_summary(dec_store["st"], channel, as_of)
            est = pol.estimate_step(resp, s0, new_budget, S, ev["average_d30_roas"]["forecast_d30_net_revenue"], ev["uncertainty"], pcfg)
        return {"finite_step": est["finite_step"], "lower_bound": est["lower_bound"], "target_d30_roas": ev["payback_target"]["target_d30_roas"],
                "label": est["label"], "assumptions": est["assumptions"]}

    return step_recalc


def make_planner(dec, result: dict, as_of: str, ecfg: dict, pre_obs_provider):
    decisions, policy_hash = result["decisions"], result["policy_hash"]

    def planner(channel: str, new_budget: float) -> dict:
        d = decisions[channel]
        ev, de = d["evidence"], ecfg["design"]
        baseline = d["current_budget"]
        lag = declared_reporting_lag(dec, as_of, channel)
        units = D.make_units(channel, baseline, de["units_per_channel"], D.derive_seed(de["assignment_seed_base"], "units", channel, as_of))
        win = D.windows(as_of, lag, de["pre_days"], de["treatment_day_options"][0])
        pre = pd.DataFrame(pre_obs_provider(channel, units, win))
        pt = ev["payback_target"]
        target = {"target_d30_roas": pt["target_d30_roas"], "source": {k: pt.get(k) for k in ("level", "n_cohorts", "installs", "first_install", "last_install", "d180_over_d30")}}
        est = ev["marginal_d30_roas_step"]
        model_estimate = {"finite_step_marginal_roas": est["finite_step"]["marginal_roas"], "lower_bound": est["lower_bound"]["lower"], "label": "MODEL-ESTIMATED (policy evidence)"}
        return D.build_plan(channel=channel, decision_as_of=as_of, policy_hash=policy_hash, baseline_budget=baseline, proposed_budget=r6(new_budget), target=target,
                            model_estimate=model_estimate, units=units, pre_obs=pre, lag_days=lag, cfg=ecfg)

    return planner


def make_real_context(dec, st, as_of: str, result: dict, budgets: dict, ceiling: float, ledger_version: int, pcfg: dict, fcfg: dict, ecfg: dict, pre_obs_provider) -> AgentContext:
    """`result` is the output of policy.decide_all for this review: the deterministic pipeline the harness always runs first."""
    return AgentContext(as_of=as_of, policy_result=result, budgets=dict(budgets), ceiling=float(ceiling), ledger_version=ledger_version, pcfg=pcfg, fcfg=fcfg, ecfg=ecfg,
                        planner=make_planner(dec, result, as_of, ecfg, pre_obs_provider), step_recalc=make_step_recalc({"st": st}, result, as_of, pcfg))
