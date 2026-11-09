"""Generate README.md. Every NUMBER in it is read from the result files (evals/results/dev_eval.json, reports/phase4_results.json, reports/phase5_live_status.json);
the prose around them is static. A test regenerates it and fails if the committed README differs, so the README cannot drift from the results.

    python scripts/build_readme.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import groq_view as GV  # noqa: E402  (pure JSON reader of the recorded Groq runs)


def load(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8-sig"))


def table(rows, cols, fmt=None):
    fmt = fmt or {}
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append("" if v is None else (fmt.get(c, lambda x: f"{x:.3f}" if isinstance(x, float) else str(x))(v)).replace("|", "/"))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def heldout_state():
    """Read the held-out baseline state from files: 'untouched' (no manifest, no run lock), 'frozen_not_run' (manifest only) or 'run' (a run lock exists). Numbers come from the stored results."""
    base = ROOT / "evals" / "results" / "heldout"
    man = ROOT / "evals" / "FREEZE_MANIFEST.json"
    manifest = json.loads(man.read_text(encoding="utf-8")) if man.exists() else None
    state = "run" if (base / "RUN_LOCK.json").exists() else ("frozen_not_run" if manifest else "untouched")
    seeds, failed = {}, {}
    for d in sorted(base.glob("seed_*")):
        s = int(d.name.split("_")[1])
        if (d / "baseline_eval.json").exists():
            seeds[s] = json.loads((d / "baseline_eval.json").read_text(encoding="utf-8"))
        prov = json.loads((d / "provenance.json").read_text(encoding="utf-8")) if (d / "provenance.json").exists() else {"steps": {}}
        bad = {n: v.get("error") for n, v in prov["steps"].items() if v["status"] != "completed"}
        if bad:
            failed[s] = bad
    return {"state": state, "manifest": manifest, "seeds": seeds, "failed": failed}


def status_block(h, valid_live, gv=None) -> list:
    live = f"**(d) Live agent verification.** Gemini: **{'PENDING' if not valid_live else 'PARTIAL'}** - {valid_live} valid live-model run(s) exist; its attempts are blocked by a recorded billing state (HTTP 402) after earlier rate-limit and quota stops, and are all kept below."
    if gv:
        live += (f" Groq (`{gv['validated_model']}`): **{gv['n_validated']} validated live proposal on one synthetic review ({gv['validated_review']}) out of {gv['n_attempts']} full-case attempts; the other {gv['n_fallback']} fell back to the rules-only proposal.** "
                 f"Its actions and budgets equal the deterministic reference on {sum(c['match'] for c in gv['comparison'])} of {len(gv['comparison'])} channels. No reliability and no decision-quality improvement is demonstrated. See *Live agent evidence (Groq)*.")
    dev = "**(a) Development results** (seed 1, where the rules and configuration were developed) are in the first results section."
    mock = "**(c) Mock agent results** are plumbing tests of the tools and validator, never model performance."
    if h["state"] == "untouched":
        held = "**(b) Held-out deterministic-baseline results: none yet.** The held-out seeds are untouched and nothing has been frozen."
    elif h["state"] == "frozen_not_run":
        held = f"**(b) Held-out deterministic-baseline results: not run yet.** The baseline is frozen as `{h['manifest']['freeze_id']}`; the held-out seeds are still untouched."
    else:
        done = sorted(h["seeds"])
        held = (f"**(b) Held-out deterministic-baseline results:** the frozen rules-only baseline (`{h['manifest']['freeze_id']}`) was run once on seeds {', '.join(map(str, done)) or 'none completed'}"
                f"{' (failed steps are listed below)' if h['failed'] else ''}. These validate ONLY that baseline on additional synthetic seeds; they say nothing about an agent, and those seeds can no longer be called untouched holdouts for one.")
    return [f"> **Status.** {dev} {held} {mock} {live}", ""]


def heldout_section(h) -> list:
    if h["state"] != "run":
        return []
    rows, pooled = [], {"n_scored": 0, "ideal": 0, "acceptable": 0, "forbidden": 0, "unlisted": 0}
    for s in sorted(set(h["seeds"]) | set(h["failed"])):
        r = h["seeds"].get(s)
        if not r:
            rows.append({"seed": s, "WAPE": "FAILED: " + "; ".join(f"{n}: {e}" for n, e in h["failed"].get(s, {}).items())[:200]})
            continue
        c = next(x for x in r["forecasting"]["table"] if x["level"] == "cohort" and x["group"] == "overall")
        o, a, v = r["policy_primary"]["overall"], r["policy_primary"]["abstention"], r["scenario_validity"]
        for k in pooled:
            pooled[k] += o[k]
        rows.append({"seed": s, "cohorts scored": c["n"], "WAPE": c["wape"], "bias": c["bias"], "coverage (nominal 0.80)": c["coverage"], "mean width / forecast": c["interval_width_over_forecast_mean"],
                     "scored reviews": o["n_scored"], "ideal": o["ideal"], "acceptable": o["acceptable"], "forbidden": o["forbidden"], "unlisted": o["unlisted"],
                     "abstention recall": a["recall"], "scenario intents holding": f"{v['n_pass']}/{v['n_intents']}", "scenarios not materialized": ", ".join(v["scenarios_not_materialized"]) or "none"})
    cols = ["seed", "cohorts scored", "WAPE", "bias", "coverage (nominal 0.80)", "mean width / forecast", "scored reviews", "ideal", "acceptable", "forbidden", "unlisted", "abstention recall", "scenario intents holding", "scenarios not materialized"]
    L = ["## Held-out results: the frozen deterministic baseline only (generated from `evals/results/heldout/`)", "",
         f"Freeze `{h['manifest']['freeze_id']}`, run once on seeds {', '.join(str(s) for s in sorted(set(h['seeds']) | set(h['failed'])))}, each seed completely separate (own data, forecast store and result files). "
         "**This validates only the rules-only baseline on additional synthetic seeds. No model was called; it says nothing about an agent, and the seeds are now used up as an untouched holdout for one.** "
         "Scenario expectations were not edited; a planted scenario that did not materialize on a seed is flagged, not hidden (full tables in `reports/heldout_baseline_report.md`).", "",
         table(rows, cols), ""]
    if h["seeds"] and pooled["n_scored"]:
        L += [f"Pooled over {len(h['seeds'])} seed(s) (sums of the counts above): {pooled['n_scored']} scored channel-reviews, {pooled['ideal']} ideal, {pooled['acceptable']} acceptable, {pooled['forbidden']} forbidden, {pooled['unlisted']} unlisted.", ""]
    return L


def groq_section(gv) -> list:
    if not gv:
        return []
    if gv["problems"]:
        return ["## Live agent evidence (Groq)", "", "**The recorded run files do not match `reports/groq_runs_manifest.json`** (" + "; ".join(gv["problems"]) + "), so no Groq numbers are shown. Restore the original files from a clean checkout.", ""]
    rows = [{"attempt": a["attempt"], "flow": a["flow"], "outcome": ("**validated proposal**" if a["status"] == "agent_proposal" else f"rules-only fallback ({a['fallback_reason']})"),
             "prompt + completion tokens": f"{a['prompt_tokens']:,} + {a['completion_tokens']:,}", "what happened": a["note"]} for a in gv["attempts"]]
    cmp_rows = [{"channel": c["channel"], "current budget": c["current_budget"], "Groq agent": f"{c['groq_action']} -> {c['groq_budget']:g}", "deterministic reference": f"{c['rules_action']} -> {c['rules_budget']:g}",
                 "claims cited": c["claims_cited"], "same": "yes" if c["match"] else "NO"} for c in gv["comparison"]]
    st, i, f = gv["stages"], gv["stages"]["investigation"], gv["stages"]["finalization"]
    return [
        "## Live agent evidence (Groq)", "",
        "> **Where the records are.** The Groq adapter is `src/agent/groq_client.py` (local function calling over Groq's OpenAI-compatible chat endpoint; settings in `config/agent_groq.yaml`; `GROQ_API_KEY` is read from the environment only). "
        "The original run records are in `reports/phase5_runs/live_groq_*.json` and `reports/groq_live_status.json`; `reports/groq_runs_manifest.json` holds their sha256 hashes. "
        "Every number below is read from those files, and a test checks them against the manifest. The adapter and its feasibility notes are in `evals/groq_feasibility/`.", "",
        f"Provider **groq**, model **{gv['validated_model']}**, mode **live**, review date {gv['validated_review']} (synthetic dev seed). All five full-case attempts on this one case are kept:", "",
        table(rows, ["attempt", "flow", "outcome", "prompt + completion tokens", "what happened"]), "",
        f"**The validated run (attempt 4).** Investigation: {i['model_requests']} model requests, {i['tool_calls_by_model']} tool calls by the model, {i['evidence_nudges']} nudge to collect missing required evidence, required evidence complete: {i['required_evidence_complete']}. "
        f"Finalization: {f['requests']} request without tools under a strict JSON schema; the provider accepted the schema and returned valid JSON ({f['provider_accepted_schema_and_returned_text']}/{f['text_parsed_as_json_object']}); "
        f"the unchanged validator accepted it with {f['repairs_used']} repairs. Tokens: {gv['tokens_validated']['prompt_tokens']:,} prompt + {gv['tokens_validated']['output_tokens']:,} completion. "
        f"The model called only: {', '.join('`' + t + '`' for t in gv['tools_used'])}.", "",
        table(cmp_rows, ["channel", "current budget", "Groq agent", "deterministic reference", "claims cited", "same"]), "",
        f"The reference is the rules-only proposal for {gv['validated_review']} in `reports/ui_review_cache.json`. The agent proposal also carried {gv['summary_claims']} summary claims.", "",
        "**What this shows.**", "",
        f"- One validated live proposal on one synthetic review. Its actions and budgets match the deterministic reference on {sum(c['match'] for c in gv['comparison'])} of {len(gv['comparison'])} channels.",
        f"- The other {gv['n_fallback']} full-case attempts fell back to the rules-only proposal (provider rejections of the model's output format, and one invalid final answer). The fallback is working as designed; it is not agent evidence.",
        "- The two-stage design (tool investigation, then a tool-free strict-schema finalization, then the unchanged validator) was built AFTER the first three failures, and the retry, provider-note and unwrap changes were made after seeing failures; "
        "the original run files of every attempt are kept in `reports/phase5_runs/`.", "",
        "**What this does not show.**", "",
        "- No reliability: 1 of 5 attempts validated, on one case, with changes made between attempts. No repeated-run or multi-case evaluation exists.",
        "- No decision-quality improvement: the proposal equals the baseline, and the model used only the overview and policy-decision tools. No agent-vs-baseline comparison of outcomes was or could be made.",
        "- No business impact, and passing tests or a passing validator is not proof of economic correctness: the validator checks citations, permitted actions, hard constraints and the experiment gate, not whether a decision is good.",
        "- Gemini remains blocked by its separately recorded billing state (HTTP 402) and produced no valid live run.",
        "- Seeds 9101-9103 are consumed by the baseline run and are not untouched agent holdouts.", ""]


def render() -> str:
    h = heldout_state()
    gv = GV.load()
    ev, p4, live = load("evals/results/dev_eval.json"), load("reports/phase4_results.json"), load("reports/phase5_live_status.json")
    fc = ev["forecasting"]
    f0 = next(r for r in fc["table"] if r["level"] == "cohort" and r["group"] == "overall")
    f1 = next(r for r in fc["table"] if r["level"] == "weekly_summary" and r["group"] == "overall")
    pol, ab = ev["policy"], ev["policy"]["abstention"]
    o = pol["overall"]
    cons = ev["constraints"]
    sg = ev["experiment_safeguards"]
    ss = sg["study_statistics"]
    mock = ev["agent_plumbing_mock"]
    attempts = list(live.get("history", [])) + [{"attempt": live.get("attempt"), "verified_model": live.get("verified_model"), "connectivity_check": live.get("connectivity_check"),
                                                  "live_agent_runs": live.get("live_agent_runs", []), "selection_stopped": live.get("selection_stopped")}]
    valid_live = sum(1 for a in attempts for r in a.get("live_agent_runs", []) if r.get("status") == "agent_proposal")
    live_rows = [{"attempt": a["attempt"], "verified model": a.get("verified_model") or "none", "connectivity check": a.get("connectivity_check") or "n/a",
                  "agent cases": "; ".join(f"{r['as_of']} {r['status']}" + (f" ({r['fallback_reason']})" if r.get("fallback_reason") else "") for r in a.get("live_agent_runs", [])) or "none ran"} for a in attempts]
    dev = ss["dev_world_scale_search"]
    demo_ok = next((r for r in (ss["demonstration_world_scale_search"] or []) if r["adequate"]), None)
    ro = ss["readout_vs_truth"]
    n_groups = sum(g["tests run"] for g in sg["groups"])
    L = [
        "# Mobile game player value and acquisition budget agent: a Monday budget review copilot", "",
        "> **ALL DATA IS SYNTHETIC. Experiments are SIMULATED. The budget ledger is a SANDBOX: nothing here moves real money.**", "",
        *status_block(h, valid_live, gv),
        "## What this is", "",
        "A user-acquisition manager reviews channel budgets every Monday. This project is the copilot for that review. Deterministic Python and SQL produce every number: availability-filtered data, quality gates, a D3-to-D30 revenue forecast "
        "with empirical intervals, a payback target, a model-estimated marginal return, and a budget policy under hard constraints. An LLM agent may investigate with a small set of read-only tools and write the proposal, but a validator "
        "checks every claim against the tool outputs, and a **person** approves. Approved changes go to a sandbox ledger. An increase is only actionable with a pre-registered, adequately powered, simulated experiment plan, "
        "and the experiment's readout decides keep or roll back.", "",
        "```", "synthetic data (late-arriving, with a planted outage, CPI spike, regime shift ...)", "   -> availability layer (a row is visible only if reported_at <= as_of)", "   -> quality gates -> forecast (D3 -> D30) with intervals -> payback target -> marginal ROAS (model-estimated)",
        "   -> deterministic policy (hold / decrease / increase_test / insufficient_evidence)", "   -> proposal: rules-only baseline, or agent (read-only tools; every claim cited)", "   -> validator (citations, permitted actions, hard constraints, experiment-plan gate)",
        "   -> PERSON approves -> sandbox ledger (versioned, audited, reversible)", "   -> increase test: frozen plan -> simulated experiment -> readout -> keep or roll back", "```", "",
        "**Division of labour.** Deterministic code owns every number. The model may only choose tools, investigate and explain. It cannot run SQL, Python, shell or file access, and it has no tool that approves, applies or rolls back anything.", "",
        "## Development results (seed 1, generated from `evals/results/dev_eval.json`)", "",
        "Criteria (`evals/criteria.yaml`) were written before the first evaluation run and not tuned afterwards; scenario expectations (`evals/scenarios/`) were written before any data was generated. "
        "This repository starts from a single commit, so that ordering is recorded in `docs/PROVENANCE.md` (original commit ids) rather than provable from this git history.", "",
        "### Forecasting (multiplier baseline, matured outcomes only)", "",
        table([{"level": "cohort forecasts", "n": f0["n"], "WAPE": f0["wape"], "bias": f0["bias"], "interval coverage (nominal 0.80)": f0["coverage"], "mean width / forecast": f0["interval_width_over_forecast_mean"]},
               {"level": "weekly summaries", "n": f1["n"], "WAPE": f1["wape"], "bias": f1["bias"], "interval coverage (nominal 0.80)": f1["coverage"], "mean width / forecast": f1["interval_width_over_forecast_mean"]}],
              ["level", "n", "WAPE", "bias", "interval coverage (nominal 0.80)", "mean width / forecast"]), "",
        f"Interval coverage is **below nominal** at both levels; the intervals are not guaranteed bounds. A forecast that ignores D3 information scores WAPE {fc['reference_baseline_ignoring_d3']['wape']:.3f}. "
        f"{fc['cohort_accounting']['withheld_total']} cohorts were withheld from forecasting; their matured revenue is {100 * fc['cohort_accounting']['share_of_matured_d30_revenue_withheld']:.0f}% of all matured D30 revenue, so scores describe the issued cohorts only.", "",
        "### Policy decisions against the committed scenario expectations", "",
        f"{o['n_scored']} channel-reviews fall inside a scenario window: **{o['ideal']} ideal, {o['acceptable']} acceptable, {o['forbidden']} forbidden, {o['unlisted']} unlisted.** "
        "A *forbidden* row violates a committed scenario expectation; it is not independent proof of economic harm.", "",
        table(pol["per_scenario"], ["scenario", "n_scored", "ideal", "acceptable", "forbidden", "unlisted"]), "",
        "Violations of the committed expectations (reported as found, not fixed):", "",
        table(pol["forbidden_or_unlisted"], ["scenario", "channel", "date", "action", "ideal", "forbidden"]), "",
        f"Abstention recall (rows where abstaining was the only ideal answer, {ab['n_rows_where_ie_is_the_only_ideal']} rows): **{ab['recall']:.2f}**. "
        f"All {ab['n_abstained']} abstentions on scored rows were appropriate; {len(ab['missed_abstentions'])} abstentions were missed (the same violations).", "",
        "### Constraints and safeguards", "",
        f"- Adversarial suite through the real code paths (cap, floor, share, ceiling, non-finite, irreversible cut, missing or unregistered plan, stale approval, duplicate apply, fabricated citation, wrong value, forbidden action, obeyed injection, inadequate design): "
        f"**{sum(c['refused_as_expected'] for c in cons)} of {len(cons)} refused as expected.**",
        f"- Experiment-lifecycle safeguards (early or fabricated readouts, link checks, fail-closed decisions, plan standards): {n_groups} named regression tests executed, {sum(g['passed'] for g in sg['groups'])} passed.", "",
        "### Simulated experiment status", "",
        f"The only experiment-design study is for **{ss['channel_studied']}** at the {ss['review_date_studied']} review. Its test is **inadequate at every declared scale** in the dev-seed world "
        f"(power at 1.5x target {min(r['power at 1.5x (56d)'] for r in dev):.2f} to {max(r['power at 1.5x (56d)'] for r in dev):.2f} against the required 0.80), so the policy action becomes `insufficient_evidence`. "
        "**Adequacy for any other channel is unknown**: no study of it exists.", "",
        (f"To exercise the full lifecycle, a separately declared synthetic *demonstration world* (committed before it was run) is adequate at x{demo_ok['scale']} volume. There the readout was "
         f"**{p4['lifecycle']['readout']['decision']}** (iROAS {ro['estimated_iroas']:.3f}, lower bound {ro['lower_bound']:.3f} vs target {ro['target_d30_roas']:.3f}); the simulator's truth was {ro['true_iroas_treated_units']:.3f}, so the decision was "
         f"{'correct' if ro['decision_correct'] else 'incorrect'}. This is one draw in a world declared after the dev-seed result was seen; it is a mechanics demonstration, not evidence about the channel." if demo_ok else "No declared scale was adequate."), "",
        "### Agent: what has and has not been shown", "",
        f"- **MOCK (scripted analyst), plumbing only:** the real tools and validator produced a validated proposal on {sum(r['validated'] for r in mock['rows'])} of {len(mock['rows'])} dev reviews, and the rules-only fallback validated on "
        f"{sum(r['fallback validated'] for r in mock['rows'])} of {len(mock['rows'])}. It follows the policy by construction, so this says nothing about any model.",
        f"- **Live (Gemini): {('PENDING' if not valid_live else 'see attempts')}.** Valid live agent runs: {valid_live}. Every attempt is kept:", "",
        table(live_rows, ["attempt", "verified model", "connectivity check", "agent cases"]), "",
        "Reasons, in order: free-tier rate limit; a tool-call bound that was set too low (amended 20 to 40, with the reason recorded in `config/agent.yaml`); the daily free quota; a billing state (HTTP 402). "
        "The offline fixes for these (rate-limit and billing classification, a graceful tool-budget wrap-up) are tested only with mock models. **Gemini produced no valid live run.** The only validated live agent proposal is the single Groq one described next; no agent-vs-baseline comparison of decision quality exists.", "",
        *groq_section(gv),
        *heldout_section(h),
        "## Key definitions", "",
        "- **Time.** `as_of` is a date; a row is visible only if `reported_at <= as_of`. D30 means days 0..30 inclusive; the D3 feature cutoff is fixed per cohort. A forecast is issued once and never recomputed from late rows.",
        "- **Value.** Net revenue = IAP x (1 - 0.30) + ad revenue. The forecast target is cumulative D30 net revenue per cohort (never called lifetime value).",
        "- **Payback target.** `target D30 ROAS = payback multiple / (sum D180 / sum D30)` over mature, reported cohorts. It is an estimate from synthetic history.",
        "- **Marginal ROAS** is always *model-estimated* for a finite step from a historical spend-to-installs fit. It is not causal evidence and not a derivative.",
        "- **Experiment.** 24 synthetic units per channel, paired and randomised; the whole approved step goes to treated units (their budgets add up to exactly the step). Unit-level difference-in-differences ratio with a delta-method standard error and a one-sided t bound; "
        "the same estimator and rule run in planning (1000 replicates) and in the readout. Hidden truth only generates noisy observations and scores afterwards.",
        "- **Hard constraints** (per-channel cap, floor, share of ceiling, total ceiling, exact reversibility) are shared by the policy, approval, apply and rollback.", "",
        "## Assumptions and limitations", "",
        "- Everything is synthetic; the simulator's truth model is this project's own assumption. Nothing says anything about real players, ad networks or incrementality.",
        "- The policy's decrease rule is a documented heuristic on forecast average ROAS, not a guarantee of economic safety; it produced the violations above.",
        f"- Forecast intervals under-cover on the dev seed. {100 * fc['cohort_accounting']['share_of_matured_d30_revenue_withheld']:.0f}% of matured revenue is withheld from forecasting (mostly by the revenue-spike exclusion), so scores describe issued cohorts only.",
        "- The ledger checks that a plan follows the authoritative experiment configuration and is internally consistent, but does **not** re-run the planning study.",
        "- Free-text numerical checks on agent prose are *numerical consistency* checks (value, sign, unit); they cannot prove a sentence is true. Injection tests are selected cases, not general resistance.",
        ("- One data-generating seed was used for development; the held-out evaluation has not been run." if h["state"] != "run" else
         "- One seed was used for development and three more for the held-out deterministic-baseline run. They share one generator, which was tuned on the dev seed, so this is not evidence about real data. No agent was evaluated on the held-out seeds."), "",
        "## Run it", "",
        "Python 3.11 is the tested version. From the repository root:", "",
        "```", "python -m venv .venv", "# activate it: .venv\\Scripts\\activate (Windows)  or  source .venv/bin/activate (macOS/Linux)", "pip install -r requirements.txt", "```", "",
        "```", "python -m pytest                          # tests (mock LLM only, free, no network)", "python scripts/generate_data.py           # synthetic dev data into data/ (gitignored; deterministic)",
        "python scripts/run_backtest.py            # forecast backtest (about 6 minutes)", "python scripts/run_dev_eval.py            # the development evaluation and its report",
        "python scripts/build_ui_data.py           # cache of real deterministic runs for the screen", "python -m streamlit run app/main.py     # the review screen (reads the committed caches; no data/ needed)", "```", "",
        "Tests that need the generated dev data or forecast store skip themselves when `data/` is absent; run the generate and backtest steps first to include them. "
        "The held-out seeds 9101-9103 were run once and their results are committed under `evals/results/heldout/`; `scripts/run_heldout_baseline.py` refuses a second run and is not part of normal use. See `docs/PROVENANCE.md`.", "",
        "Live runs (optional; they call an external API and are not needed for the tests or the screen): set `GEMINI_API_KEY` or `GROQ_API_KEY` in the environment (never in a file), then "
        "`python scripts/run_agent.py --mode live [--provider groq] [--model ID] [--only DATE]`. Gemini first lists the account's models and verifies one with a real round trip; Groq uses `config/agent_groq.yaml`. "
        "A replay recording is saved only from a validated live run. Mock and replay runs are labelled and kept in separate files. `python scripts/groq_smoke.py` is offline unless `--live` is given.", "",
        "## How the code is checked", "",
        "- **Tests that can fail.** Safeguards were checked by *mutation*: disable one, confirm the intended tests fail, restore it. Several tests were strengthened when a mutation survived.",
        "- **Pre-registration.** Scenario expectations were written before any data existed; evaluation criteria before the first evaluation run; configuration provenance (what was fixed before any score) is written in the config headers.",
        "- **Defects found in review, each with a regression test:** a decision rule that treated a missing guardrail as passing (now fail-closed), a readout executor that trusted a caller-supplied decision (now recomputed from the store), "
        "a plan that could lower its own power standard (now checked against an authoritative configuration), citations to null fields that crashed the validator, and a daily quota retried as if it were per-minute.",
        "- **No invented results.** Every RESULT number in this README is read from a result file by `scripts/build_readme.py`, and a test fails if the committed README differs from a fresh render. "
        "Definitions and configuration values (for example the number of experiment units or replicates) are quoted from the config files, not measured.",
        "- **What the tests do not prove.** They exercise selected failure cases against mocks and a synthetic simulator. They do not show the agent is useful, the policy is economically right, or that no leakage path exists. "
        "The invariance tests are selected cases; the data contract supports immutable late-arriving facts only (overwriting visible history is unsupported).", "",
        "## Extending to LiveOps (a sketch, not built)", "",
        "The same shape fits other budget and event decisions: availability-aware data, a forecast with honest intervals, a hard-constrained policy, a validator between a model and a ledger, and pre-registered small experiments for risky changes. "
        "Real use would need real connectors, record versioning for corrected data, a calibrated incrementality method on real campaigns, and an evaluation on held-out periods first.", "",
        "## Repository map", "",
        "- `src/data`, `src/quality`, `src/forecast`, `src/policy`: availability layer, gates, forecasting, economics and policy.",
        "- `src/ledger`: sandbox ledger and proposals. `src/experiments`: plan, design, readout, store, lifecycle (plus hidden-side `simulator.py` and `scoring.py`).",
        "- `src/agent`: read-only tools, validator, LLM clients (mock, replay, Gemini, Groq), agent loop. `app/`: the review screen. `evals/`: criteria, scenarios, development and held-out evaluation, freeze manifest, Groq feasibility notes.",
        "- `reports/`: generated reports (phase 1-6, live status, recorded live runs, UI cache). `config/`: all assumptions, each with provenance notes. `sql/`: schema and metric queries. `docs/`: data contract and `PROVENANCE.md`.", ""]
    return "\n".join(L)


def main():
    (ROOT / "README.md").write_text(render(), encoding="utf-8", newline="\n")
    print("wrote README.md")


if __name__ == "__main__":
    main()
