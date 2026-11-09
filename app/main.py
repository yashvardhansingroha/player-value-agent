"""Monday budget review (copilot): ONE screen over real deterministic runs on the synthetic dev seed. ALL DATA IS SYNTHETIC; experiments are SIMULATED; the ledger is a SANDBOX.

What it shows: the review table, forecasts with intervals, quality gates, a proposal with its evidence and validator result, approve/apply/reject/rollback in a sandbox ledger,
experiment status and the dev evaluation. What it does NOT do: call any model, read any API key, or touch hidden data. The agent can only propose; a PERSON (you, here) approves.
Run with:  .venv\\Scripts\\python.exe -m streamlit run app\\main.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app import ui_logic as U  # noqa: E402


def table(df, where=st):
    """Every table fills its container and hides the index."""
    where.dataframe(df, hide_index=True, width="stretch")

st.set_page_config(page_title="Monday Budget Review", layout="wide")
st.warning("SYNTHETIC DATA  |  SIMULATED EXPERIMENTS  |  SANDBOX LEDGER: nothing on this screen is real revenue, retention or spend, and nothing here moves real money.")
st.title("Monday budget review (copilot)")

live = U.live_status()
held = U.heldout_view()
st.info(U.status_text(live, held, U.groq_summary()) + " The rules-only baseline is deterministic code. Approval is always a person.")

cache = U.load_cache()
ok, msg = U.cache_status(cache)
if not ok:
    st.error(msg)
    st.stop()

dates = U.review_dates(cache)
with st.sidebar:
    st.header("Review")
    day = st.selectbox("Monday review date (dev seed)", dates, index=dates.index("2025-10-13") if "2025-10-13" in dates else 0, key="day")
    source = st.radio("Proposal source", list(U.SOURCES), format_func=lambda k: U.SOURCES[k], key="source")
    approver = st.text_input("Approver (a person)", "reviewer", key="approver")
    if st.button("Reset sandbox ledger", key="reset"):
        st.session_state.sandbox = U.new_sandbox(cache)
        st.session_state.messages = []
    st.caption("The sandbox ledger starts from the dev seed's sandbox budgets. Proposals are pre-computed against that starting ledger, so after one is applied the others are refused as stale: that is the ledger's safeguard, not a bug.")
    st.caption("This screen never asks for, reads or stores an API key.")

if "sandbox" not in st.session_state:
    st.session_state.sandbox = U.new_sandbox(cache)
st.session_state.setdefault("messages", [])
led = U.open_ledger(st.session_state.sandbox)  # one connection per run
src = U.get_source(cache, day, source)
prop = src["proposal"]

t1, t2, t3, t4, t5, t6 = st.tabs(["1. Monday review", "2. Forecasts and quality", "3. Proposal and evidence", "4. Approval and ledger", "5. Experiments (SIMULATED)", "6. Evaluation"])

with t1:
    s = led.state()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Review date", day)
    c2.metric("Sandbox total budget / day", f"{s['total']:.1f}")
    c3.metric("Ceiling", f"{s['ceiling']:.1f}")
    c4.metric("Reserve (unallocated)", f"{s['reserve']:.1f}")
    st.subheader("What the deterministic policy says, channel by channel")
    table(U.review_table(cache, day))
    ft = U.forecast_table(cache, day).set_index("channel")
    st.subheader("Average D30 ROAS vs the payback target")
    st.bar_chart(ft[["avg D30 ROAS", "payback target"]])
    st.caption("Average ROAS is the weekly forecast of the INCLUDED cohorts only (withheld cohorts are not in it). The payback target is estimated from mature synthetic history. A channel below target is a reason to look, not a proof.")

with t2:
    st.subheader("Weekly forecasts with 80% intervals (included cohorts only)")
    table(U.forecast_table(cache, day))
    st.caption("Intervals are empirical and NOMINALLY 80%. On this dev data observed coverage is below that (see tab 6): never read them as guaranteed bounds. Marginal ROAS is MODEL-ESTIMATED for a finite step, not causal.")
    ch = st.selectbox("Channel", list(cache["dates"][day]["channels"]), key="channel")
    v = cache["dates"][day]["channels"][ch]
    st.subheader(f"Evidence gates for {ch}")
    table(U.stages_table(cache, day, ch))
    if v["forecast"]:
        f = v["forecast"]
        st.write(f"Window {f['window'][0]} to {f['window'][1]}: {f['included_cohorts']} of {f['n_expected_dates']} install dates are in the forecast; {f['excluded_withheld_cohorts']} were withheld "
                 f"(included cohorts hold {100 * (f['included_spend_share'] or 0):.0f}% of the window's spend).")
    st.write("Policy reasons: " + "; ".join(v["reasons"]))

with t3:
    st.subheader(src["title"])
    for label in src["labels"]:
        if label.startswith("MODE: MOCK"):
            st.warning(label)
        elif label.startswith(("INADEQUATE EXPERIMENT", "TOOL BUDGET", "RULES-ONLY FALLBACK", "REPLAY DRIFT")):
            st.warning(label)
        elif label.startswith("APPROVAL REQUIRED"):
            st.caption(label)
        else:
            st.info(label)
    st.caption(src["note"])
    val = src["validation"] or {}
    if val.get("ok"):
        st.success("Validator: OK (traceable evidence, permitted actions, hard constraints and the experiment-plan gate). This does not make the decision good.")
    else:
        st.error(f"Validator errors: {[e['code'] for e in val.get('errors', [])]}")
    table(U.proposal_items_table(prop))
    names = [i["channel"] for i in prop["items"]]
    pick = st.selectbox("Item", names, key="item")
    item = next(i for i in prop["items"] if i["channel"] == pick)
    st.write(item["reason"])
    claims = U.claims_table(item)
    if len(claims):
        st.markdown("**Cited claims** (each points at a tool call, a field and a population; checked by the validator):")
        table(claims)
    if src["tool_calls"]:
        st.markdown("**Evidence trail** (every tool call of the run):")
        table(U.trail_table(src))
        ids = [c["tool_call_id"] for c in src["tool_calls"] if c["status"] == "ok"]
        if ids:
            tc = st.selectbox("Show the raw result of a tool call", ids, key="toolcall")
            st.json(next(e for e in src["evidence_trail"] if e["tool_call_id"] == tc)["result"])

with t4:
    st.subheader("Approval: a person decides; the ledger re-checks everything")
    pid = prop["proposal_id"]
    status = U.proposal_status(led, pid)
    st.write(f"Proposal `{pid}` (built on ledger version {prop['ledger_version']}); status in this sandbox ledger: **{status or 'not submitted'}**; ledger head: version {led.head()}.")
    b1, b2, b3, b4 = st.columns(4)
    results = []
    if b1.button("Submit to ledger", key="submit"):
        results.append(U.submit(led, prop, day))
    if b2.button("Approve", key="approve"):
        results.append(U.decide(led, pid, "approve", approver, day))
    if b3.button("Apply", key="apply"):
        results.append(U.decide(led, pid, "apply", approver, day))
    if b4.button("Reject", key="reject"):
        results.append(U.decide(led, pid, "reject", approver, day))
    for r in results:
        st.session_state.messages.append((r["ok"], r["message"]))
    if results:
        st.rerun()  # a ledger action changed state that tabs above have already drawn: redraw everything so no number lags one action behind (messages persist in the session)
    for ok_, m in st.session_state.messages[-6:]:
        (st.success if ok_ else st.error)(m)
    st.caption("An increase test is refused unless its adequate experiment plan is registered in the ledger and bound to this exact step. A proposal with only holds has nothing to apply.")
    st.subheader("Sandbox ledger")
    cA, cB = st.columns(2)
    table(U.ledger_budgets(led), cA)
    table(U.ledger_versions(led), cB)
    changes = U.active_changes(led)
    if changes:
        cid = st.selectbox("Applied change to roll back", [c["change_id"] for c in changes], key="rollback_pick")
        if st.button("Roll back this change", key="rollback"):
            r = U.rollback(led, cid, approver, day)
            st.session_state.messages.append((r["ok"], r["message"]))
            st.rerun()
    else:
        st.caption("No applied change to roll back yet.")
    with st.expander("Audit log"):
        table(U.audit_table(led))

with t5:
    ev = U.experiment_view()
    st.subheader("Experiment status (everything here is SIMULATED)")
    if not ev:
        st.info("reports/phase4_results.json is missing.")
    else:
        st.write(f"The ONLY experiment-design study that exists is for **{ev['channel']}** at the {ev['as_of']} review. In the dev-seed world its test is INADEQUATE at every declared scale. "
                 "**Adequacy for any other channel (for example meta) is UNKNOWN: no study of it exists.** No increase in this project has an adequate plan on the dev seed, so no increase test is actionable here.")
        st.markdown("Dev-seed world: power at 1.5x target and false-keep at the target boundary, by scale-up of volume")
        table(ev["dev_world"][["scale", "adequate", "power at 1.5x (56d)", "false-keep at 1.0x (56d)", "median SE of iROAS (56d, at 1.5x)"]])
        st.markdown(f"Separately declared DEMONSTRATION world (not the dev-seed channel): first adequate scale x{ev['chosen_scale']}")
        table(ev["demo_world"][["scale", "adequate", "power at 1.5x (56d)", "false-keep at 1.0x (56d)"]])
        if ev["readout"]:
            r = ev["readout"]
            st.markdown(f"Demonstration lifecycle (SIMULATED), plan `{ev['plan_hash'][:16]}...`: readout decision **{r['decision']}** (`{r['decision_code']}`); iROAS {r['estimate']['iroas']:.3f}, lower bound {r['estimate']['lower_bound']:.3f} "
                        f"vs target {r['rule']['target_d30_roas']:.3f}. Scored afterwards against the simulator's truth: {ev['score']['true_iroas_treated_units']:.3f}.")
            table(ev["transitions"][["step", "ledger version", "kind", "total", "reserve", "sim clock"]])
    increases = [i for i in prop["items"] if i["action"] == "increase_test"]
    st.write("Increase tests in the selected proposal: " + (", ".join(f"{i['channel']} (plan {(i['experiment_plan_hash'] or 'none')[:12]})" for i in increases) if increases else "none."))

with t6:
    ev = U.evaluation_view()
    st.subheader("(a) Development evaluation (seed 1, where the rules were developed; not held-out evidence)")
    if not ev:
        st.info("evals/results/dev_eval.json is missing; run scripts/run_dev_eval.py.")
    else:
        o = ev["policy_overall"]
        st.write(f"Policy vs the committed scenario expectations: {o['n_scored']} scored channel-reviews: {o['ideal']} ideal, {o['acceptable']} acceptable, **{o['forbidden']} forbidden**, {o['unlisted']} unlisted. "
                 "A forbidden row violates a committed scenario expectation; it is not independent proof of economic harm.")
        table(ev["per_scenario"])
        st.markdown("Violations of the committed scenario expectations (reported as found, not fixed):")
        table(ev["violations"])
        f = ev["forecast_overall"]
        st.write(f"Forecasting (cohorts): WAPE {f['wape']:.3f}, bias {f['bias']:+.3f}, interval coverage {f['coverage']:.3f} against nominal 0.80 (UNDER nominal), mean interval width/forecast {f['interval_width_over_forecast_mean']:.3f}.")
        st.write(f"Constraint suite: {int(ev['constraints']['refused_as_expected'].sum())} of {len(ev['constraints'])} adversarial cases refused as expected.")
        table(ev["constraints"])
    st.subheader("(b) Held-out evaluation of the frozen deterministic baseline")
    if held["state"] != "run":
        st.write("No held-out result yet: " + ("nothing has been frozen." if held["state"] == "untouched" else f"frozen as {held['freeze_id']}, not run."))
    else:
        st.write(f"Freeze {held['freeze_id']}, each seed run once and kept separate. This validates ONLY the rules-only baseline on additional synthetic seeds; no model was called, nothing is said about an agent, "
                 "and a planted scenario that did not materialize on a seed is flagged here rather than hidden.")
        if len(held["rows"]):
            table(held["rows"])
        for sd, bad in held["failed"].items():
            st.warning(f"Seed {sd}: step(s) did not complete (preserved, not retried): " + "; ".join(f"{n}: {e}" for n, e in bad.items()))
    st.subheader("(c) Mock agent plumbing")
    st.write("Scripted-analyst runs through the real tools and validator test the plumbing only; they are not evidence about any model.")
    st.subheader("(d) Live agent verification: Gemini " + live["status"] + " (blocked by a recorded billing state, HTTP 402; attempts preserved as found, none is agent evidence)")
    groq = U.groq_summary()
    if groq:
        st.write(f"Groq ({groq['validated_model']}): {groq['n_validated']} validated live proposal on one synthetic review ({groq['validated_review']}) out of {groq['n_attempts']} full-case attempts; the other {groq['n_fallback']} fell back to the rules-only proposal. "
                 "The validated proposal equals the deterministic reference on every channel; no reliability and no decision-quality improvement is demonstrated. Details: the README section Live agent evidence.")
        table(pd.DataFrame(groq["attempts"])[["attempt", "flow", "status", "fallback_reason", "prompt_tokens", "completion_tokens", "note"]])
    rows = [{"attempt": a.get("attempt"), "verified model": a.get("verified_model"), "connectivity": a.get("connectivity_check"),
             "agent cases": "; ".join(f"{r['as_of']} {r['status']}" + (f" ({r['fallback_reason']})" if r.get("fallback_reason") else "") for r in a.get("live_agent_runs", [])) or "none ran"} for a in live["attempts"]]
    table(pd.DataFrame(rows))
    st.caption("Four kinds of result are kept apart: (a) development, (b) held-out deterministic baseline, (c) mock agent plumbing, (d) live agent verification (Gemini blocked; one Groq case validated).")
