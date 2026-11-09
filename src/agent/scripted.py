"""A SCRIPTED analyst for MOCK demonstrations: no LLM. It investigates with the real tools and writes a cited proposal that follows the deterministic
reference. It exists to show the plumbing (tools -> evidence -> validator -> proposal) on real review data. Its output is NEVER evidence about any
model or agent. ALL DATA IS SYNTHETIC."""
import json

from src.agent.context import AgentContext
from src.agent.llm import LLMTurn, MockLLM, ToolCall, text_turn


def _claim(tc, field, value, population, channel):
    return {"tool_call_id": tc, "field": field, "value": value, "population": population, "channel": channel}


def scripted_llm(ctx: AgentContext) -> MockLLM:
    calls = [("get_review_overview", {})]
    for ch in ctx.channels:
        calls.append(("get_policy_decision", {"channel": ch}))
    for ch in ctx.channels:
        d = ctx.decision(ch)
        if any(not s["passed"] for s in d["stages"]):
            calls.append(("get_quality_checks", {"channel": ch}))
        if d["action"] in ("hold", "decrease") and d["evidence"].get("average_d30_roas", {}).get("point") is not None:
            calls.append(("get_economics", {"channel": ch}))
        if d["action"] == "increase_test":
            calls.append(("plan_experiment", {"channel": ch, "new_budget": round(d["proposed_budget"], 6)}))
    ids = {(name, a.get("channel"), a.get("new_budget")): f"tc_{n:03d}" for n, (name, a) in enumerate(calls, 1)}
    items = []
    for ch in ctx.channels:
        d, ref = ctx.decision(ch), ctx.reference(ch)
        pid = ids[("get_policy_decision", ch, None)]
        claims = [_claim(pid, "action", d["action"], "policy_decision", ch)]
        item = {"channel": ch, "action": ref["action"], "new_budget": ref["budget"], "qualification": "none", "experiment_plan_hash": None, "claims": claims}
        failed = [s for s in d["stages"] if not s["passed"]]
        if d["action"] == "increase_test":
            tc = ids[("plan_experiment", ch, round(d["proposed_budget"], 6))]
            q = ctx.qualification(ch, d["proposed_budget"])
            claims.append(_claim(tc, "qualifies", bool(q["qualifies"]), "experiment_plan_for_exact_step", ch))
            pa = q["plan"]["power_assessment"]
            if q["qualifies"]:
                item.update(qualification="experiment_backed", experiment_plan_hash=q["plan_hash"])
                item["reason"] = f"The policy proposes a bounded increase test from {ctx.budgets[ch]:g} to {ref['budget']:g}; the deterministic recalculation and an adequate experiment plan qualify it."
            else:
                shown = pa["chosen_treatment_days"] or max(int(k) for k in pa["per_duration"])
                alt = next(a for a in pa["per_duration"][str(shown)]["alternatives"] if abs(a["multiple"] - pa["standards"]["alternative_multiple_of_target"]) < 1e-9)
                power, need = round(alt["keep_rate"], 3), pa["standards"]["target_power"]
                claims += [_claim(tc, "plan.power_at_alternative", power, "frozen_experiment_plan_planning_study", ch),
                           _claim(tc, "plan.target_power", need, "frozen_experiment_plan_planning_study", ch)]
                item["reason"] = (f"The policy proposed an increase test, but the experiment plan for this exact step is inadequate: simulated power {power} is below the required {need}. "
                                  "The increase cannot be tested, so the evidence is insufficient and the budget is unchanged.")
        elif failed and ("get_quality_checks", ch, None) in ids:
            tc = ids[("get_quality_checks", ch, None)]
            names = [s["stage"] for s in d["stages"]]
            i = names.index(failed[0]["stage"])
            claims += [_claim(tc, f"stages.{i}.stage", failed[0]["stage"], "weekly_window_quality_checks", ch), _claim(tc, f"stages.{i}.passed", False, "weekly_window_quality_checks", ch)]
            item["reason"] = f"Stage {failed[0]['stage']} did not pass (it blocks: {failed[0]['blocks']}); I follow the policy action {d['action']}."
        elif ("get_economics", ch, None) in ids:
            tc = ids[("get_economics", ch, None)]
            avg, tgt = d["evidence"]["average_d30_roas"]["point"], d["evidence"].get("payback_target", {}).get("target_d30_roas")
            claims.append(_claim(tc, "average_d30_roas.point", round(avg, 3), "included_cohorts_forecast_average", ch))
            text = f"Policy action {d['action']}: forecast average D30 ROAS {round(avg, 3)} (included cohorts)"
            if tgt is not None:
                claims.append(_claim(tc, "payback_target.target_d30_roas", round(tgt, 3), "mature_history_payback_target", ch))
                text += f" against the payback target {round(tgt, 3)}"
            item["reason"] = text + "."
        else:
            item["reason"] = f"I follow the deterministic policy action {d['action']} for this channel."
        items.append(item)
    summary = ("Scripted MOCK analyst: every action follows the deterministic policy after the experiment-design gate; this demonstrates the tools, citations and validator, "
               "not any model's judgement.")
    return MockLLM([LLMTurn(tool_calls=[ToolCall(n, dict(a), f"mock-{i}") for i, (n, a) in enumerate(calls)]), text_turn(json.dumps({"summary": summary, "summary_claims": [], "items": items}))])
