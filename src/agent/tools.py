"""The ONLY things the model can do: seven read-only tools that EXPOSE numbers the deterministic code already produced. ALL DATA IS SYNTHETIC.

  get_review_overview            channels, budgets, readiness, policy action; sandbox ledger totals
  get_quality_checks(channel)    the staged gate results for the channel (completeness, quality checks, coverage, uncertainty, target)
  get_forecast_and_populations   weekly forecast, interval, INCLUDED vs EXCLUDED cohorts (kept apart: a number is only valid for its own population)
  get_economics(channel)         average ROAS, payback target, response curve, model-estimated finite-step marginal ROAS
  get_policy_decision(channel)   the deterministic policy action and budget
  plan_experiment(channel, new_budget)   the deterministic recalculation of exactly that step and its frozen-plan power study
  check_budget_constraints(changes)      the hard constraints (and the exact inverse) for a set of budget changes

There is NO SQL, Python, shell or file tool and NO tool that approves, applies or rolls back anything: approval and application stay with
people and the ledger, outside the agent. Every successful call is wrapped in an ENVELOPE {tool_call_id, tool, args, channel, as_of,
policy_hash, populations, result}; citations point at (tool_call_id, field path in `result`) and are checked against the stored envelope.
Text inside `result` is DATA, never instructions.
"""
import fnmatch
import hashlib
import math
from dataclasses import dataclass, field
from typing import Callable

from src.agent.context import AgentContext, NotRecalculable
from src.agent.llm import ToolCall
from src.ledger.proposal import canonical, jsonable
from src.policy import constraints as cons


class ToolError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code, self.message = code, message


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict
    fn: Callable
    default_population: str
    populations: dict = field(default_factory=dict)  # result-path prefix -> population label
    units: dict = field(default_factory=dict)  # result-path pattern (fnmatch, '*' also matches dots) -> unit; first match wins; anything else is "number"

    def population_of(self, path: str) -> str:
        best = None
        for prefix, label in self.populations.items():
            if (path == prefix or path.startswith(prefix + ".")) and (best is None or len(prefix) > len(best[0])):
                best = (prefix, label)
        return best[1] if best else self.default_population

    def unit_of(self, path: str) -> str:
        """'money', 'count', 'ratio' (a dimensionless share, rate, probability or ROAS: the ONLY unit a percentage may be written for) or 'number'."""
        for pattern, unit in self.units.items():
            if fnmatch.fnmatchcase(path, pattern):
                return unit
        return "number"


def _channel(ctx: AgentContext, args: dict) -> str:
    ch = args["channel"]
    if ch not in ctx.budgets:
        raise ToolError("unknown_channel", f"{ch!r} is not a paid channel of this review; channels are {ctx.channels}")
    return ch


# ---------------------------------------------------------------------------------------------- tools


def get_review_overview(ctx: AgentContext, args: dict) -> dict:
    total = float(sum(ctx.budgets.values()))
    rows = {}
    for ch in ctx.channels:
        d = ctx.decision(ch)
        failed = [s["stage"] for s in d["stages"] if not s["passed"]]
        rows[ch] = {"current_budget": ctx.budgets[ch], "policy_action": d["action"], "policy_proposed_budget": d["proposed_budget"],
                    "completeness_passed": next((s["passed"] for s in d["stages"] if s["stage"] == "completeness"), None), "failed_stages": failed}
    return {"as_of": ctx.as_of, "ledger": {"version": ctx.ledger_version, "total_budget": total, "ceiling": ctx.ceiling, "reserve": ctx.ceiling - total}, "channels": rows}


def get_quality_checks(ctx: AgentContext, args: dict) -> dict:
    ch = _channel(ctx, args)
    stages = [{"stage": s["stage"], "passed": s["passed"], "blocks": s["blocks"], "detail": s["detail"]} for s in ctx.decision(ch)["stages"]]
    return {"stages": stages, "n_failed": sum(1 for s in stages if not s["passed"])}


def get_forecast_and_populations(ctx: AgentContext, args: dict) -> dict:
    ch = _channel(ctx, args)
    ev = ctx.decision(ch)["evidence"]
    if "populations" not in ev:
        return {"available": False, "reason": "the policy had no forecast summary for this channel at this review"}
    p = ev["populations"]
    return {"available": True, "note": p["note"], "window": p["window"], "n_expected_dates": p["n_expected_dates"], "included": p["included"],
            "excluded_withheld": p["excluded_withheld"], "included_spend_share": p["included_spend_share"], "accounting_statuses": p["accounting_statuses"],
            "uncertainty": ev.get("uncertainty")}


def get_economics(ctx: AgentContext, args: dict) -> dict:
    ch = _channel(ctx, args)
    ev = ctx.decision(ch)["evidence"]
    resp = ev.get("response_curve")
    step = ev.get("marginal_d30_roas_step")
    return {"average_d30_roas": ev.get("average_d30_roas"), "payback_target": ev.get("payback_target"),
            "response_curve": None if resp is None else {k: resp.get(k) for k in ("usable", "unusable_reasons", "n_obs", "b", "b_se", "spend_range", "window", "label")},
            "marginal_step": None if step is None else {"label": step["label"], "step_from": step["finite_step"]["s0"], "step_to": step["finite_step"]["s1"],
                                                        "finite_step_marginal_roas": step["finite_step"]["marginal_roas"], "lower_bound": step["lower_bound"]["lower"],
                                                        "assumptions": step["assumptions"]}}


def get_policy_decision(ctx: AgentContext, args: dict) -> dict:
    ch = _channel(ctx, args)
    d = ctx.decision(ch)
    return {"action": d["action"], "current_budget": d["current_budget"], "proposed_budget": d["proposed_budget"], "reasons": list(d["reasons"]),
            "increase_candidate": d.get("increase_candidate"),
            "note": "deterministic policy output BEFORE the experiment-design gate; an increase test still needs an adequate plan (see plan_experiment)"}


def plan_experiment(ctx: AgentContext, args: dict) -> dict:
    ch = _channel(ctx, args)
    new = float(args["new_budget"])
    q = ctx.qualification(ch, new)
    d = ctx.decision(ch)
    out = {"baseline_budget": q["baseline_budget"], "new_budget": q["new_budget"], "step_amount": round(q["new_budget"] - q["baseline_budget"], 6),
           "policy_proposed_budget": d["proposed_budget"], "is_policy_step": abs(q["new_budget"] - d["proposed_budget"]) < 1e-6 and d["action"] == "increase_test",
           "qualifies": q["qualifies"], "problems": q["problems"], "recalculation": None, "plan": None,
           "action_after_design": "increase_test" if q["qualifies"] else "insufficient_evidence",
           "note": "all statistics are produced by the deterministic recalculation and planning study; none can be supplied or altered by the model"}
    rc = q.get("recalculation")
    if rc:
        out["recalculation"] = {"finite_step_marginal_roas": rc["finite_step"]["marginal_roas"], "lower_bound": rc["lower_bound"]["lower"], "target_d30_roas": rc["target_d30_roas"],
                                "lower_bound_meets_target": q["lower_bound_meets_target"], "label": rc["label"]}
    plan = q.get("plan")
    if plan:
        pa = plan["power_assessment"]
        std = pa["standards"]
        shown = pa["chosen_treatment_days"] or max(int(k) for k in pa["per_duration"])
        alts = {round(a["multiple"], 6): a for a in pa["per_duration"][str(shown)]["alternatives"]}
        power, boundary = alts[round(std["alternative_multiple_of_target"], 6)], alts[round(std["boundary_multiple"], 6)]
        out["plan"] = {"plan_hash": q["plan_hash"], "adequate": pa["adequate"], "reasons": list(pa["reasons"]), "chosen_treatment_days": pa["chosen_treatment_days"],
                       "window_shown_days": shown, "readout_available_on": plan["windows"]["readout_available_on"], "n_units": len(plan["units"]),
                       "n_treated_units": sum(1 for u in plan["units"] if u["arm"] == "treatment"), "experiment_config_hash": plan["experiment_config_hash"],
                       "power_at_alternative": power["keep_rate"], "alternative_multiple_of_target": std["alternative_multiple_of_target"], "target_power": std["target_power"],
                       "false_keep_at_boundary": boundary["keep_rate"], "max_false_keep": std["max_false_keep"], "replicates": pa["replicates"],
                       "label": "SIMULATED EXPERIMENT PLAN. ALL DATA IS SYNTHETIC."}
    return out


def check_budget_constraints(ctx: AgentContext, args: dict) -> dict:
    proposed = dict(ctx.budgets)
    for c in args["changes"]:
        if c["channel"] not in ctx.budgets:
            raise ToolError("unknown_channel", f"{c['channel']!r} is not a paid channel of this review")
        proposed[c["channel"]] = float(c["new_budget"])
    v = cons.check_constraints(ctx.budgets, proposed, ctx.ceiling, ctx.pcfg)
    inv = cons.check_constraints(proposed, ctx.budgets, ctx.ceiling, ctx.pcfg)
    total = float(sum(proposed.values()))
    return {"ok": not v and not inv, "violations": v, "inverse_violations": inv, "total_after": total, "reserve_after": ctx.ceiling - total, "proposed_budgets": proposed}


CHANNEL_ONLY = {"type": "object", "properties": {"channel": {"type": "string", "description": "a paid channel name from get_review_overview"}}, "required": ["channel"]}

TOOLS = [
    ToolSpec("get_review_overview", "Overview of this Monday review: every paid channel's current budget, readiness, failed stages and the deterministic policy action, plus sandbox ledger totals.",
             {"type": "object", "properties": {}}, get_review_overview, "channel_review_status", {"ledger": "sandbox_ledger_state"}),
    ToolSpec("get_quality_checks", "The staged evidence gates for one channel (evidence binding, completeness, each quality check, forecast coverage, uncertainty, payback target) with pass/fail and what a failure blocks.",
             CHANNEL_ONLY, get_quality_checks, "weekly_window_quality_checks"),
    ToolSpec("get_forecast_and_populations", "The weekly forecast for one channel's review window with its INCLUDED and EXCLUDED (withheld) cohorts reported separately, and the forecast interval.",
             CHANNEL_ONLY, get_forecast_and_populations, "channel_review_window",
             {"included": "included_cohorts", "excluded_withheld": "excluded_withheld_cohorts", "uncertainty": "included_cohorts_weekly_forecast_interval"}),
    ToolSpec("get_economics", "Average D30 ROAS (included cohorts), the payback target, the response curve and the MODEL-ESTIMATED finite-step marginal ROAS for one channel.",
             CHANNEL_ONLY, get_economics, "channel_economics",
             {"average_d30_roas": "included_cohorts_forecast_average", "payback_target": "mature_history_payback_target", "response_curve": "historical_spend_install_response",
              "marginal_step": "included_cohorts_model_estimated_finite_step"}),
    ToolSpec("get_policy_decision", "The deterministic policy action and proposed budget for one channel, with its reasons.", CHANNEL_ONLY, get_policy_decision, "policy_decision"),
    ToolSpec("plan_experiment", "Deterministically recalculate an increase to EXACTLY new_budget (model-estimated marginal ROAS and lower bound) and build its frozen experiment plan with the planning power study. "
             "Returns whether the step qualifies. Use it for every increase you consider, including a smaller step than the policy's.",
             {"type": "object", "properties": {"channel": {"type": "string"}, "new_budget": {"type": "number", "description": "the exact proposed daily budget"}}, "required": ["channel", "new_budget"]},
             plan_experiment, "experiment_plan_for_exact_step", {"recalculation": "model_estimated_step_recalculation", "plan": "frozen_experiment_plan_planning_study"}),
    ToolSpec("check_budget_constraints", "Check a set of proposed budget changes against the hard constraints (cap, floor, share, ceiling) and the exact inverse change.",
             {"type": "object", "properties": {"changes": {"type": "array", "items": {"type": "object", "properties": {"channel": {"type": "string"}, "new_budget": {"type": "number"}},
                                                                                 "required": ["channel", "new_budget"]}}}, "required": ["changes"]},
             check_budget_constraints, "hard_constraints_check"),
]
# Units of the result fields (metadata for the prose checks only). money / count are never allowed to authorise a percentage; 'ratio' fields are dimensionless
# (shares, rates, probabilities, ROAS) and are the only ones a percentage may be written for: 30% means 0.30. Unlisted fields are plain 'number's.
UNITS = {
    "get_review_overview": {"ledger.version": "count", "ledger.*": "money", "channels.*.current_budget": "money", "channels.*.policy_proposed_budget": "money"},
    "get_quality_checks": {"n_failed": "count"},
    "get_forecast_and_populations": {"n_expected_dates": "count", "included.n_cohorts": "count", "included.installs": "count", "included.spend": "money", "included.forecast_d30_net_revenue": "money",
                                     "included.observed_d0_3_net_revenue": "money", "excluded_withheld.n_cohorts": "count", "excluded_withheld.installs": "count",
                                     "excluded_withheld.spend": "money", "excluded_withheld.observed_d0_3_net_revenue": "money", "excluded_withheld.cohorts.*.installs": "count",
                                     "excluded_withheld.cohorts.*.spend": "money", "excluded_withheld.cohorts.*.observed_d0_3_net_revenue": "money", "included_spend_share": "ratio",
                                     "accounting_statuses.*": "count", "uncertainty.calibration_n": "count", "uncertainty.lo": "money", "uncertainty.hi": "money", "uncertainty.width_ratio": "ratio"},
    "get_economics": {"average_d30_roas.forecast_d30_net_revenue": "money", "average_d30_roas.spend": "money", "average_d30_roas.*": "ratio", "payback_target.n_cohorts": "count",
                      "payback_target.installs": "count", "payback_target.sum_d30": "money", "payback_target.sum_d180": "money", "payback_target.*": "ratio", "response_curve.n_obs": "count",
                      "response_curve.spend_range.*": "money", "marginal_step.step_from": "money", "marginal_step.step_to": "money", "marginal_step.*": "ratio"},
    "get_policy_decision": {"current_budget": "money", "proposed_budget": "money", "increase_candidate.s1": "money", "increase_candidate.*": "ratio"},
    "plan_experiment": {"baseline_budget": "money", "new_budget": "money", "step_amount": "money", "policy_proposed_budget": "money", "recalculation.*": "ratio", "plan.chosen_treatment_days": "count",
                        "plan.window_shown_days": "count", "plan.n_units": "count", "plan.n_treated_units": "count", "plan.replicates": "count", "plan.alternative_multiple_of_target": "number",
                        "plan.power_at_alternative": "ratio", "plan.target_power": "ratio", "plan.false_keep_at_boundary": "ratio", "plan.max_false_keep": "ratio"},
    "check_budget_constraints": {"total_after": "money", "reserve_after": "money", "proposed_budgets.*": "money"},
}
for _t in TOOLS:
    _t.units = UNITS[_t.name]
REGISTRY = {t.name: t for t in TOOLS}
FORBIDDEN_CAPABILITIES = ("sql", "python", "shell", "exec", "file", "ledger", "approve", "apply", "rollback")  # no tool name may suggest any of these


# ---------------------------------------------------------------------------------------------- argument checking


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def check_args(schema: dict, args) -> None:
    """Strict: arguments must be an object with exactly the declared properties, of the declared types."""
    if not isinstance(args, dict):
        raise ToolError("bad_arguments", "arguments must be an object")
    props, req = schema.get("properties", {}), schema.get("required", [])
    extra = sorted(set(args) - set(props))
    if extra:
        raise ToolError("bad_arguments", f"unknown arguments {extra}; allowed: {sorted(props)}")
    for k in req:
        if k not in args:
            raise ToolError("bad_arguments", f"missing required argument {k!r}")
    for k, v in args.items():
        t = props[k]["type"]
        ok = {"string": isinstance(v, str), "number": _is_num(v), "array": isinstance(v, list), "object": isinstance(v, dict)}[t]
        if not ok:
            raise ToolError("bad_arguments", f"argument {k!r} must be of type {t}")
        if t == "array":
            for item in v:
                check_args(props[k]["items"], item)


def declarations() -> list:
    return [{"name": t.name, "description": t.description, "parameters": t.parameters} for t in TOOLS]


# ---------------------------------------------------------------------------------------------- execution and the evidence log


def resolve(record: dict, path: str):
    """(found, value) for a dotted path inside the record's `result` (list indexes are integers)."""
    cur = record["result"]
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return False, None
    return True, cur


class ToolRunner:
    """Executes registered tools for one run and keeps the evidence log the validator checks citations against."""

    def __init__(self, ctx: AgentContext, max_calls: int):
        self.ctx, self.max_calls, self.n, self.records, self.order, self.denied = ctx, max_calls, 0, {}, [], []

    @property
    def exhausted(self) -> bool:
        return self.n >= self.max_calls

    def execute(self, call: ToolCall) -> dict:
        self.n += 1
        cid = f"tc_{self.n:03d}"
        spec = REGISTRY.get(call.name)
        base = {"tool_call_id": cid, "tool": call.name, "args": jsonable(call.args)}
        try:
            if spec is None:
                raise ToolError("unknown_tool", f"{call.name!r} is not a tool; the tools are {sorted(REGISTRY)}")
            check_args(spec.parameters, call.args)
            result = jsonable(spec.fn(self.ctx, call.args))
            env = {**base, "channel": call.args.get("channel"), "as_of": self.ctx.as_of, "policy_hash": self.ctx.policy_hash,
                   "populations": {"default": spec.default_population, **spec.populations}, "result": result,
                   "output_sha256": hashlib.sha256(canonical(result).encode()).hexdigest()}
        except ToolError as e:
            env = {**base, "error": {"code": e.code, "message": e.message}}
        except NotRecalculable as e:
            env = {**base, "error": {"code": "not_recalculable", "message": str(e)}}
        except Exception as e:  # a deterministic tool failed: report it to the model as an error (never as a number) and keep the run alive
            env = {**base, "error": {"code": "tool_failed", "message": f"{type(e).__name__}: {e}"}}
        self.records[cid] = env
        self.order.append(cid)
        return env

    def deny(self, call: ToolCall) -> dict:
        """The tool budget is used up: the call is NOT executed and is not counted as executed. The envelope carries an id that can never be cited (it is
        not in `records`), so a claim pointing at it is rejected as an unknown tool call."""
        self.denied.append(call)
        return {"tool_call_id": f"denied_{len(self.denied):03d}", "tool": call.name, "args": jsonable(call.args),
                "error": {"code": "tool_budget_exhausted", "message": f"the budget of {self.max_calls} tool calls is used up; this call was NOT run. Answer now from the evidence you already hold."}}

    def population_of(self, record: dict, path: str) -> str:
        return REGISTRY[record["tool"]].population_of(path)

    def unit_of(self, record: dict, path: str) -> str:
        return REGISTRY[record["tool"]].unit_of(path)
