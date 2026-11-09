"""Everything the tools and the validator read for ONE Monday review. ALL DATA IS SYNTHETIC; the ledger is a SANDBOX.

The harness runs the deterministic pipeline FIRST (`policy_result` = policy.decide_all) and hands this context to both the tools (which
only expose it) and the validator (which independently re-derives what the agent is allowed to propose). The agent cannot reach the
stores, the ledger or the hidden truth: it only sees what the registered tools return.

Two recalculation hooks are injected (real ones live in planning.py; tests pass small fakes):
  step_recalc(channel, new_budget) -> model-estimated finite-step marginal ROAS and lower bound for exactly that step
  planner(channel, new_budget)     -> the frozen-before-outcomes experiment plan, power study included, for exactly that step
Plans are never invented by the model: a plan exists only if the deterministic planner produced it.
"""
from dataclasses import dataclass, field
from typing import Callable

from src.experiments import plan as P


class NotRecalculable(Exception):
    """The deterministic code cannot evaluate this step (for example the policy never evaluated an increase for the channel)."""


def r6(x) -> float:
    return round(float(x), 6)


@dataclass
class AgentContext:
    as_of: str
    policy_result: dict  # policy.decide_all output
    budgets: dict  # sandbox ledger budgets at the review (read-only copy)
    ceiling: float
    ledger_version: int
    pcfg: dict
    fcfg: dict
    ecfg: dict
    planner: Callable
    step_recalc: Callable
    _plans: dict = field(default_factory=dict)
    _recalcs: dict = field(default_factory=dict)

    @property
    def policy_hash(self) -> str:
        return self.policy_result["policy_hash"]

    @property
    def channels(self) -> list:
        return sorted(self.budgets)

    def decision(self, channel: str) -> dict:
        return self.policy_result["decisions"][channel]

    # ------------------------------------------------------------------ deterministic recalculation (cached per exact step)
    def recalc_for(self, channel: str, new_budget: float) -> dict:
        key = (channel, r6(new_budget))
        if key not in self._recalcs:
            self._recalcs[key] = self.step_recalc(channel, r6(new_budget))
        return self._recalcs[key]

    def plan_for(self, channel: str, new_budget: float) -> dict:
        key = (channel, r6(new_budget))
        if key not in self._plans:
            self._plans[key] = self.planner(channel, r6(new_budget))
        return self._plans[key]

    def all_plans(self) -> dict:
        return {P.plan_hash(p): p for p in self._plans.values()}

    def qualification(self, channel: str, new_budget: float) -> dict:
        """Can an increase of `channel` to exactly `new_budget` be proposed? Needs (1) the step recalculated by the deterministic code with
        a lower bound at or above the target and (2) an ADEQUATE frozen plan for exactly this step. Both are recomputed here, never taken
        from the model."""
        new_budget = r6(new_budget)
        cur = self.budgets[channel]
        out = {"channel": channel, "baseline_budget": cur, "new_budget": new_budget, "qualifies": False, "problems": []}
        if not new_budget > cur:
            out["problems"].append("not an increase")
            return out
        try:
            rc = self.recalc_for(channel, new_budget)
        except NotRecalculable as e:
            out["problems"].append(f"the step cannot be recalculated: {e}")
            return out
        out["recalculation"] = rc
        out["lower_bound_meets_target"] = bool(rc["lower_bound"]["lower"] >= rc["target_d30_roas"])
        if not out["lower_bound_meets_target"]:
            out["problems"].append(f"lower bound {rc['lower_bound']['lower']:.4f} is below the target {rc['target_d30_roas']:.4f} at this step")
        plan = self.plan_for(channel, new_budget)
        out["plan"], out["plan_hash"] = plan, P.plan_hash(plan)
        out["plan_adequate"] = bool(plan["power_assessment"]["adequate"])
        if not out["plan_adequate"]:
            out["problems"].append("the experiment design is inadequate: " + "; ".join(plan["power_assessment"]["reasons"]))
        out["qualifies"] = out["lower_bound_meets_target"] and out["plan_adequate"]
        return out

    def reference(self, channel: str) -> dict:
        """The deterministic reference for this channel: the policy decision after the experiment-design gate. An increase test whose
        design is inadequate becomes insufficient_evidence with the budget unchanged (Phase 4 rule)."""
        d = self.decision(channel)
        cur = self.budgets[channel]
        ref = {"channel": channel, "policy_action": d["action"], "policy_budget": r6(d["proposed_budget"]), "current_budget": cur,
               "action": d["action"], "budget": r6(d["proposed_budget"]), "design_adjusted": False}
        if d["action"] == "increase_test":
            q = self.qualification(channel, d["proposed_budget"])
            ref["qualification"] = {k: v for k, v in q.items() if k != "plan"}
            if not q["qualifies"]:
                ref.update(action="insufficient_evidence", budget=cur, design_adjusted=True)
        return ref
