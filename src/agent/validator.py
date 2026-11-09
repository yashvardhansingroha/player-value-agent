"""The proposal validator. ALL DATA IS SYNTHETIC; the ledger is a SANDBOX. It is AUTHORITATIVE and does not trust the model.

Two layers (the same first layer also checks the rules-only fallback, so the fallback keeps every gate):
  1. validate_items     structure, permitted action substitutions, budgets, hard constraints and the exact inverse, and the experiment-plan
                        gate. It re-derives the reference from the deterministic pipeline itself (AgentContext.reference/qualification),
                        never from what the model fetched or claimed. An increase is accepted only if the step was recalculated by the
                        deterministic code with a lower bound at or above target AND its frozen plan is adequate and correctly bound.
  2. validate_citations (agent output only) every claim must point at an existing, successful tool call of THIS run by (tool_call_id,
                        field path) and match on value, channel, as-of date, configuration (policy hash) and POPULATION; a matching number
                        from another cohort, channel or call is rejected. Every number written in the prose must be a cited value.

Permitted substitutions (reference = policy decision after the experiment-design gate):
    increase_test (qualified)  -> increase_test with a step no larger than the policy's, hold, insufficient_evidence
    decrease                   -> a decrease no larger than the policy's, hold, insufficient_evidence
    hold                       -> hold, insufficient_evidence
    insufficient_evidence      -> insufficient_evidence only
Passing these checks is not proof that the proposal is good; it establishes that it is traceable, within the policy and within the hard constraints.
"""
import json
import math
import re
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, ValidationError

from src.agent.context import AgentContext, r6
from src.agent.tools import ToolRunner, resolve
from src.experiments import plan as P
from src.ledger.proposal import Proposal, ProposalItem, make_proposal_id
from src.policy import constraints as cons

TOL = 1e-6
ALLOWED = {"increase_test": {"increase_test", "hold", "insufficient_evidence"}, "decrease": {"decrease", "hold", "insufficient_evidence"},
           "hold": {"hold", "insufficient_evidence"}, "insufficient_evidence": {"insufficient_evidence"}}


# ---------------------------------------------------------------------------------------------- the model's output schema


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool_call_id: str
    field: str  # dotted path inside that tool call's `result`
    value: Union[bool, float, str]
    population: str  # the population label the tool attached to that field
    channel: Optional[str] = None


class AgentItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    channel: str
    action: Literal["increase_test", "hold", "decrease", "insufficient_evidence"]
    new_budget: float
    qualification: Literal["none", "experiment_backed"] = "none"
    experiment_plan_hash: Optional[str] = None
    reason: str
    claims: list[Claim]


class AgentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str
    summary_claims: list[Claim] = []
    items: list[AgentItem]


def parse_output(text) -> tuple:
    """(AgentOutput or None, errors). Accepts plain JSON or JSON inside a code fence; anything else is malformed."""
    if text is None or not str(text).strip():
        return None, [{"code": "malformed_output", "detail": "the model returned no text"}]
    s = str(text).strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL)
    if m:
        s = m.group(1).strip()
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j < i:
        return None, [{"code": "malformed_output", "detail": "no JSON object found in the reply"}]
    try:
        raw = json.loads(s[i:j + 1])
    except json.JSONDecodeError as e:
        return None, [{"code": "malformed_output", "detail": f"invalid JSON: {e}"}]
    try:
        return AgentOutput.model_validate(raw), []
    except ValidationError as e:
        return None, [{"code": "schema_invalid", "detail": f"{'.'.join(map(str, err['loc']))}: {err['msg']}"} for err in e.errors()[:6]]


# ---------------------------------------------------------------------------------------------- numbers


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def decimals_of(text: str) -> int:
    return len(text.split(".")[1]) if "." in text else 0


def number_matches(claimed: float, actual: float, decimals: int) -> bool:
    """`claimed` (written with `decimals` decimals) must be the CORRECT ROUNDING of `actual`, not coarser than 2% / 0.005 away from it, and have the SAME SIGN
    (a negative value is never matched by a positive one, however close to zero)."""
    if not (_is_num(claimed) and _is_num(actual)):
        return False
    if claimed * actual < 0:
        return False
    if math.isclose(claimed, actual, rel_tol=1e-9, abs_tol=1e-12):
        return True
    return math.isclose(round(actual, decimals), claimed, abs_tol=1e-9) and abs(claimed - actual) <= max(0.005, 0.02 * abs(actual))


def value_kind(x) -> str:
    """bool | str | number | nonfinite | null | object | list | other. Only bool, str and number can be cited; nothing is ever converted to make a value fit."""
    if isinstance(x, bool):
        return "bool"
    if isinstance(x, str):
        return "str"
    if isinstance(x, (int, float)):
        return "number" if math.isfinite(x) else "nonfinite"
    return {type(None): "null", dict: "object", list: "list", tuple: "list"}.get(type(x), "other")


def value_matches(claimed, actual) -> bool:
    """Total: never raises. Different kinds never match; numbers match by `number_matches`; bools and strings by equality."""
    ck, ak = value_kind(claimed), value_kind(actual)
    if ck != ak or ck not in ("bool", "str", "number"):
        return False
    if ck != "number":
        return claimed == actual
    text = repr(float(claimed))
    return number_matches(float(claimed), float(actual), 12 if "e" in text else decimals_of(text))


_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_MINUS = "-" + chr(0x2212) + chr(0x2012) + chr(0x2013)  # hyphen-minus, Unicode minus, figure dash, en dash (written as codes so the file stays ASCII)
_NUMBER = re.compile(rf"(?<![\w.])([{_MINUS}+])?(\d{{1,3}}(?:,\d{{3}})+|\d+)(\.\d+)?(\s?%|\s+percent\b)?(?!\w)")


def numbers_in_text(text: str) -> list:
    """Every number written in prose as {value (unsigned), decimals, percent, sign ('-', '+' or None), token}, ignoring ISO dates and numbers glued to letters
    (D30, ids, hashes). A sign counts only when it directly precedes the digits and is not itself glued to a word or digit, so in '10-20' the 20 is unsigned.
    '30%' keeps percent=True: it denotes 0.30 and is NOT the same statement as 30."""
    out = []
    for m in _NUMBER.finditer(_DATE.sub(" ", text)):
        s = m.group(2).replace(",", "") + (m.group(3) or "")
        sign = None if not m.group(1) else ("+" if m.group(1) == "+" else "-")
        out.append({"value": float(s), "decimals": decimals_of(s), "percent": bool(m.group(4)), "sign": sign, "token": m.group(0)})
    return out


# ---------------------------------------------------------------------------------------------- layer 1: items, policy, constraints, plan gate


def _err(code, channel=None, detail=""):
    return {"code": code, "channel": channel, "detail": detail}


def validate_items(items: list, ctx: AgentContext) -> list:
    """`items`: dicts with channel, action, new_budget, qualification, experiment_plan_hash. Returns error dicts (empty = valid)."""
    errs = []
    chans = [i["channel"] for i in items]
    if sorted(chans) != ctx.channels:
        return [_err("channel_set_mismatch", None, f"proposal channels {sorted(chans)} must be exactly the review's paid channels {ctx.channels}, once each")]
    proposed = {}
    pc = ctx.pcfg
    for it in items:
        ch, act, new, cur = it["channel"], it["action"], it["new_budget"], ctx.budgets[it["channel"]]
        if not _is_num(new) or new < 0:
            errs.append(_err("invalid_budget", ch, f"new budget {new!r} is not a finite non-negative number"))
            continue
        proposed[ch] = float(new)
        if act in ("hold", "insufficient_evidence") and abs(new - cur) > TOL:
            errs.append(_err("budget_changed_without_action", ch, f"{act} must keep the budget at {cur}, not {new}"))
        if (act == "increase_test" and not new > cur + TOL) or (act == "decrease" and not new < cur - TOL):
            errs.append(_err("action_budget_mismatch", ch, f"{act} but the budget goes {cur} -> {new}"))
        ref = ctx.reference(ch)
        if act not in ALLOWED[ref["action"]]:
            errs.append(_err("forbidden_substitution", ch, f"the reference for {ch} is {ref['action']}"
                             f"{' (policy said ' + ref['policy_action'] + ' but the experiment design is inadequate)' if ref['design_adjusted'] else ''}; "
                             f"{act} is not a permitted substitution (allowed: {sorted(ALLOWED[ref['action']])})"))
            continue
        if act == "decrease" and new < ref["policy_budget"] - TOL:
            errs.append(_err("cut_larger_than_policy", ch, f"cut to {new} is deeper than the policy's {ref['policy_budget']}"))
        if act == "increase_test":
            errs += _increase_gate(it, ctx, ref)
        elif it.get("qualification", "none") != "none" or it.get("experiment_plan_hash"):
            errs.append(_err("unexpected_plan", ch, f"{act} must not carry a plan or an experiment qualification"))
    if len(proposed) == len(items):
        full = {**ctx.budgets, **proposed}
        for v in cons.check_constraints(ctx.budgets, full, ctx.ceiling, pc):
            errs.append(_err("constraint_violation", v.get("channel"), f"{v['constraint']}: {v['detail']}"))
        for v in cons.check_constraints(full, ctx.budgets, ctx.ceiling, pc):
            errs.append(_err("not_reversible", v.get("channel"), f"the exact inverse would violate {v['constraint']}: {v['detail']}"))
    return errs


def _increase_gate(it: dict, ctx: AgentContext, ref: dict) -> list:
    """An increase needs (1) the step recalculated by the deterministic code with lower bound >= target, (2) an ADEQUATE frozen plan for exactly this
    step, (3) the plan correctly bound to this channel, baseline, step, policy configuration and date, and (4) that plan named in the item."""
    ch, new, cur = it["channel"], r6(it["new_budget"]), ctx.budgets[it["channel"]]
    errs = []
    if new > ref["policy_budget"] + TOL:
        errs.append(_err("step_larger_than_policy", ch, f"step to {new} exceeds the policy's {ref['policy_budget']}"))
    if new - cur < ctx.pcfg["steps"]["min_step_fraction"] * cur - TOL:
        errs.append(_err("step_below_minimum", ch, f"step {new - cur:.4f} is below {100 * ctx.pcfg['steps']['min_step_fraction']:.0f}% of the budget"))
    q = ctx.qualification(ch, new)  # recomputed here, whatever the model called or claimed
    if not q["qualifies"]:
        return errs + [_err("increase_not_qualified", ch, "; ".join(q["problems"]) or "the step does not qualify")]
    if it.get("qualification") != "experiment_backed" or it.get("experiment_plan_hash") != q["plan_hash"]:
        errs.append(_err("increase_requires_bound_adequate_plan", ch, f"an increase must be 'experiment_backed' with the deterministic plan hash {q['plan_hash']} for exactly this step; "
                                                                        f"got qualification={it.get('qualification')!r}, hash={it.get('experiment_plan_hash')!r}"))
    plan = q["plan"]
    pe = P.validate_plan_structure(plan) + P.validate_plan_against_config(plan, ctx.ecfg)
    item = {"channel": ch, "current_budget": cur, "new_budget": new, "evidence": {"payback_target": {"target_d30_roas": q["recalculation"]["target_d30_roas"]}}}
    pe += P.validate_plan_binding(plan, item, {"budgets": ctx.budgets}, ctx.policy_hash, ctx.as_of)
    if pe:
        errs.append(_err("plan_invalid_or_unbound", ch, "; ".join(pe)))
    return errs


# ---------------------------------------------------------------------------------------------- layer 2: citations


def _check_claim(c: Claim, expected_channel, ctx: AgentContext, runner: ToolRunner) -> list:
    where = f"claim {c.tool_call_id}:{c.field}"
    rec = runner.records.get(c.tool_call_id)
    if rec is None:
        return [_err("unknown_tool_call", expected_channel, f"{where}: no tool call with that id was made in this run")]
    if "error" in rec:
        return [_err("tool_call_failed", expected_channel, f"{where}: that call returned an error ({rec['error']['code']}), so it is not evidence")]
    if rec["as_of"] != ctx.as_of or rec["policy_hash"] != ctx.policy_hash:
        return [_err("stale_evidence", expected_channel, f"{where}: recorded for as_of {rec['as_of']} / configuration {rec['policy_hash'][:12]}, this review is {ctx.as_of} / {ctx.policy_hash[:12]}")]
    if expected_channel is not None and c.channel != expected_channel:
        return [_err("claim_wrong_channel", expected_channel, f"{where}: the claim says channel {c.channel!r}, the item is {expected_channel!r}")]
    if rec["channel"] is not None and c.channel != rec["channel"]:
        return [_err("claim_wrong_channel", expected_channel, f"{where}: the tool call was for channel {rec['channel']!r}, the claim says {c.channel!r}")]
    if rec["channel"] is None and rec["tool"] == "get_review_overview" and c.channel is not None and not (c.field.startswith(f"channels.{c.channel}.") or c.field.startswith("ledger.")):
        return [_err("claim_wrong_channel", expected_channel, f"{where}: an overview field must be under channels.{c.channel}. or ledger.")]
    found, actual = resolve(rec, c.field)
    if not found:
        return [_err("field_not_found", expected_channel, f"{where}: no such field in that call's result")]
    ak, ck = value_kind(actual), value_kind(c.value)
    if ak not in ("bool", "str", "number"):
        return [_err("unsupported_field_type", expected_channel, f"{where}: that field is {ak if ak != 'nonfinite' else 'not a finite number'}, not a single number, text or true/false value; "
                                                                  "cite one of the scalar fields inside it, or do not cite it")]
    if ck != ak:
        return [_err("value_type_mismatch", expected_channel, f"{where}: the field holds a {ak} but the claim gives a {ck}; values are never converted to make them match")]
    label = runner.population_of(rec, c.field)
    if c.population != label:
        return [_err("wrong_population", expected_channel, f"{where}: that field describes '{label}', the claim says '{c.population}'")]
    if not value_matches(c.value, actual):
        return [_err("value_mismatch", expected_channel, f"{where}: the claim says {c.value!r}, the tool output is {actual!r}")]
    return []


def derived_entries(current: float, new: float) -> list:
    """The numbers an item's own budgets imply, with units: the budgets and the step (money), and the step as a fraction of the current budget (ratio).
    Signed forms say 'up' or 'down'; magnitude forms authorise only unsigned wording such as 'cut by 8' or 'a 10% step'."""
    step = new - current
    out = [(current, "money", False), (new, "money", False), (step, "money", False), (abs(step), "money", True)]
    if current:
        out += [(step / current, "ratio", False), (abs(step) / current, "ratio", True)]
    return out


def _text_numbers_cited(text: str, entries: list, where: str, channel) -> list:
    """NUMERICAL CONSISTENCY CHECK of free prose: every number must be a verified cited value (or derived from the item's budgets) with the same sign and
    unit. entries = [(value, unit, magnitude_only)]. A percentage is compared with ratio-unit values only, as value/100 (30% means 0.30); a money or count
    value never authorises a percentage; a negative token needs a negative value and an unsigned token a non-negative one. This does NOT establish that
    the sentences are true: it only stops a number from being quoted with a different sign, scale or unit than the evidence has."""
    errs = []
    for t in numbers_in_text(text):
        signed = -t["value"] if t["sign"] == "-" else t["value"]
        ok = False
        for value, unit, magnitude in entries:
            if magnitude and t["sign"] is not None:
                continue
            if t["percent"]:
                if unit != "ratio":
                    continue
                value = value * 100.0
            if number_matches(signed, value, t["decimals"]):
                ok = True
                break
        if not ok:
            errs.append(_err("uncited_number", channel, f"{where}: {t['token'].strip()!r} is not a verified cited value with this sign and unit"
                                                       f"{' (a percentage must be a share or ratio field: 30% means 0.30)' if t['percent'] else ''}"))
    return errs


def _cites(verified: list, runner: ToolRunner, channel: str, tool: str, field: str, value, new_budget: float = None) -> bool:
    """Is there a VERIFIED claim on `tool`'s `field` equal to `value` from a call for this channel (and, if given, for exactly this new_budget)?"""
    for c in verified:
        rec = runner.records[c.tool_call_id]
        if rec["tool"] == tool and rec["channel"] == channel and c.field == field and c.value == value:
            if new_budget is None or abs(rec["args"].get("new_budget", math.nan) - new_budget) < TOL:
                return True
    return False


def _entry(c: Claim, runner: ToolRunner) -> tuple:
    """(value, unit, False) for a VERIFIED numeric claim: the unit is the tool field's declared unit, not something the model says."""
    return float(c.value), runner.unit_of(runner.records[c.tool_call_id], c.field), False


def validate_citations(out: AgentOutput, ctx: AgentContext, runner: ToolRunner) -> list:
    errs = []
    all_numeric = []
    for it in out.items:
        ch = it.channel
        if ch not in ctx.budgets:
            continue
        verified, good = [], []
        for c in it.claims:
            e = _check_claim(c, ch, ctx, runner)
            errs += e
            if not e:
                verified.append(c)
                if value_kind(c.value) == "number":
                    good.append(_entry(c, runner))
        pol = ctx.decision(ch)["action"]
        if not _cites(verified, runner, ch, "get_policy_decision", "action", pol):
            errs.append(_err("missing_required_evidence", ch, f"every item must cite get_policy_decision.action == {pol!r} for its channel (call the tool and cite it)"))
        if it.action == "increase_test" and not _cites(verified, runner, ch, "plan_experiment", "qualifies", True, new_budget=it.new_budget):
            errs.append(_err("missing_plan_evidence", ch, "an increase must cite plan_experiment.qualifies == true from a call made for exactly this new_budget"))
        errs += _text_numbers_cited(it.reason, good + derived_entries(ctx.budgets[ch], it.new_budget), "reason", ch)
        all_numeric += good
    sc_good = []
    for c in out.summary_claims:
        e = _check_claim(c, None, ctx, runner)
        errs += e
        if not e and value_kind(c.value) == "number":
            sc_good.append(_entry(c, runner))
    money = [(float(v), "money", False) for v in [ctx.ceiling, sum(ctx.budgets.values()), *ctx.budgets.values()]]
    errs += _text_numbers_cited(out.summary, all_numeric + sc_good + money, "summary", None)
    return errs


# ---------------------------------------------------------------------------------------------- reports and proposals


def validate_agent_output(out: AgentOutput, ctx: AgentContext, runner: ToolRunner) -> dict:
    items = [{"channel": i.channel, "action": i.action, "new_budget": i.new_budget, "qualification": i.qualification, "experiment_plan_hash": i.experiment_plan_hash} for i in out.items]
    errs = validate_items(items, ctx) + validate_citations(out, ctx, runner)
    return report(errs, items, ctx)


def validate_rules_items(items: list, ctx: AgentContext) -> dict:
    """The rules-only fallback goes through the SAME gates (layer 1): substitutions, constraints, reversibility, plan gate."""
    return report(validate_items(items, ctx), items, ctx)


def report(errs: list, items: list, ctx: AgentContext) -> dict:
    plans = sorted({P.plan_hash(ctx.plan_for(i["channel"], i["new_budget"])) for i in items if i["action"] == "increase_test" and not errs})
    return {"ok": not errs, "errors": errs, "plans_to_register": plans, "note": "valid = traceable, inside the policy's permitted actions and the hard constraints; it does not make the decision good"}


def to_proposal(out: AgentOutput, ctx: AgentContext, run_labels: list) -> Proposal:
    items = []
    for i in out.items:
        items.append(ProposalItem(channel=i.channel, action=i.action, current_budget=ctx.budgets[i.channel], new_budget=r6(i.new_budget),
                                  qualification="experiment_backed" if i.action == "increase_test" else "none",
                                  experiment_plan_hash=i.experiment_plan_hash if i.action == "increase_test" else None, reason=i.reason,
                                  evidence={"claims": [c.model_dump() for c in i.claims], "run_labels": run_labels}))
    return Proposal(proposal_id=make_proposal_id(ctx.as_of, items, ctx.ledger_version, ctx.policy_hash) + "-AGENT", as_of=ctx.as_of, ledger_version=ctx.ledger_version,
                    policy_hash=ctx.policy_hash, source="agent", items=items)
