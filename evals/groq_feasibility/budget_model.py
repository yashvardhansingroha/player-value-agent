"""Token-budget model for one agent run on Groq (openai/gpt-oss-120b). Inputs: measured CHARACTER counts (stdin = measure_tokens.py output). Tokens are ESTIMATES:
tokens = chars / ratio, ratio 2.6 (pessimistic, number-heavy JSON), 3.2 (central), 3.8 (optimistic). No tokenizer for gpt-oss is available offline; the first live response's
`usage.prompt_tokens` replaces this estimate (the client logs it)."""
import json
import sys

m = json.load(sys.stdin)
RATIOS = {"pessimistic": 2.6, "central": 3.2, "optimistic": 3.8, "measured_json_4.3": 4.3, "measured_prefix_4.9": 4.9}  # the last two come from the live smoke test
TPM, TPD, RPM, RPD = 8000, 200000, 30, 1000
TOOLS_FACTOR = 1.15  # tool schemas are re-rendered by the provider's chat template; allow 15% (UNVERIFIED)
MSG_OVERHEAD = 12  # tokens of role/channel markers per message (estimate)
CALL_TOKENS = 30  # an assistant tool call message: name + short args + id (estimate)

fixed_chars = m["system_prompt_chars"] + m["task_prompt_chars"]
tools_chars = m["tool_declarations_chars_openai_wrapped"]


def rows(case):
    d = {}
    for r in m["cases"][case]["rows"]:
        d.setdefault(r["tool"], []).append(r)
    return d


def plan_sequences(case):
    """Turn-by-turn tool batches (character sizes of the model-visible envelope, 'compact' or 'full' flavour)."""
    d = rows(case)
    acts = m["cases"][case]["actions"]
    chans = m["cases"][case]["channels"]
    idx = {c: i for i, c in enumerate(chans)}
    seqs = {}
    for flavour in ("full", "compact"):
        pick = lambda tool, ch=None: d[tool][0 if ch is None or len(d[tool]) == 1 else idx[ch]][flavour]
        # FULL SWEEP (what the Gemini run did): overview, then the 4 per-channel tools for all channels (+ plan_experiment for increases)
        sweep2 = [pick("get_policy_decision", c) + pick("get_quality_checks", c) + pick("get_forecast_and_populations", c) + pick("get_economics", c) for c in chans]
        extra = [pick("plan_experiment") for c in chans if acts[c] == "increase_test" and "plan_experiment" in d]
        seqs[f"{flavour}/full_sweep"] = [[pick("get_review_overview")], sweep2 + extra, [pick("check_budget_constraints")]]
        # TARGETED: overview, policy for every channel (required by rule 4), then ONLY what the situation needs
        policy = [pick("get_policy_decision", c) for c in chans]
        need = []
        for c in chans:
            if acts[c] in ("insufficient_evidence", "decrease"):
                need.append(pick("get_quality_checks", c))
            if acts[c] in ("decrease", "hold"):
                need.append(pick("get_economics", c)) if False else None
            if acts[c] == "increase_test" and "plan_experiment" in d:
                need.append(pick("plan_experiment"))
        # one economics call for decrease channels only
        need += [pick("get_economics", c) for c in chans if acts[c] == "decrease"]
        seqs[f"{flavour}/targeted"] = [[pick("get_review_overview")], policy, need, [pick("check_budget_constraints")]]
    return seqs


def simulate(seq, ratio, out_final=1400, out_call_turn=250, max_completion=3000, repair=False):
    """Per-request prompt tokens (history is re-sent every turn) and the run total. Returns rows and totals."""
    fixed = fixed_chars / ratio + tools_chars / ratio * TOOLS_FACTOR + MSG_OVERHEAD * 3
    hist, reqs = 0.0, []
    for batch in seq:
        if not batch:
            continue
        prompt = fixed + hist
        out = out_call_turn + CALL_TOKENS * len(batch)
        reqs.append({"prompt": round(prompt), "completion": round(out), "max_completion": max_completion})
        hist += CALL_TOKENS * len(batch) + MSG_OVERHEAD + sum(c / ratio + MSG_OVERHEAD for c in batch)
    reqs.append({"prompt": round(fixed + hist), "completion": out_final, "max_completion": max_completion, "final": True})
    if repair:  # one repair: the rejected answer and the validator's errors are appended
        hist += out_final + 500
        reqs.append({"prompt": round(fixed + hist), "completion": out_final, "max_completion": max_completion, "final": True, "repair": True})
    return reqs


res = {"fixed_overhead_chars": {"system_prompt+task": fixed_chars, "tool_schemas_openai_wrapped": tools_chars}}
for case in m["cases"]:
    for name, seq in plan_sequences(case).items():
        for rn, ratio in RATIOS.items():
            reqs = simulate(seq, ratio)
            tot_in, tot_out = sum(r["prompt"] for r in reqs), sum(r["completion"] for r in reqs)
            worst = max(r["prompt"] + r["max_completion"] for r in reqs)
            res[f"{case} | {name} | {rn}"] = {"requests": len(reqs), "prompts": [r["prompt"] for r in reqs], "tokens_in": tot_in, "tokens_out": tot_out, "total": tot_in + tot_out,
                                              "worst_prompt_plus_max_completion": worst, "fits_8000_with_max_completion": worst <= TPM,
                                              "runs_per_day_by_200k": int(TPD // (tot_in + tot_out)), "min_minutes_by_tpm": round((tot_in + tot_out) / TPM, 1)}
print(json.dumps(res, indent=1))
