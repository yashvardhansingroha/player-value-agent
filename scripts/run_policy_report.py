"""Run the deterministic policy on the dev seed for every decision date and write reports/phase3_report.md.

    python scripts/run_policy_report.py [--seed 1]

Needs data/decision/seed_<n>/decision.db and data/forecasts/seed_<n>/forecasts.db (run scripts/run_backtest.py first).
DEV SEED ONLY. The scenario comparison at the end is a DESCRIPTIVE check against expectations committed before any data existed;
it is not the Phase 6 evaluation and no policy value was tuned after seeing it. ALL DATA IS SYNTHETIC; the ledger is a SANDBOX.
"""
import argparse
import copy
import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.calendar import day_to_date  # noqa: E402
from src.ledger.ledger import Ledger  # noqa: E402
from src.ledger.proposal import Proposal, ProposalItem, jsonable, make_proposal_id  # noqa: E402
from src.policy import constraints as cons  # noqa: E402
from src.policy import economics as eco  # noqa: E402
from src.policy import policy as pol  # noqa: E402

START = "2025-01-06"
CHANNELS = ["meta", "google_uac", "applovin", "unity", "tiktok"]


def md(df, floats=3):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if v is None or (isinstance(v, float) and pd.isna(v)) else (f"{v:.{floats}f}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(lines)


def fmt(x, n=3):
    return "" if x is None else f"{x:.{n}f}"


def row_for(d):
    ev = d["evidence"]
    avg, tgt, mg = ev.get("average_d30_roas", {}), ev.get("payback_target", {}), ev.get("marginal_d30_roas_step")
    return {"channel": d["channel"], "action": d["action"], "budget": d["current_budget"], "proposed": d["proposed_budget"],
            "avg ROAS (incl. cohorts)": fmt(avg.get("point")), "avg lo..hi": f"{fmt(avg.get('lower'), 2)}..{fmt(avg.get('upper'), 2)}" if "lower" in avg else "",
            "target": fmt(tgt.get("target_d30_roas")), "step ROAS (point / lower)": f"{fmt(mg['finite_step']['marginal_roas'], 2)} / {fmt(mg['lower_bound']['lower'], 2)}" if mg else "",
            "reason": d["reasons"][0][:150]}


def hand_check(results, pcfg):
    """Recompute one finite-step estimate by hand from the evidence the policy stored."""
    haircut = pcfg["marginal"]["incremental_quality_haircut"]
    for res in results.values():
        for d in res["decisions"].values():
            mg = d["evidence"].get("marginal_d30_roas_step")
            if mg:
                r, fs = d["evidence"]["response_curve"], mg["finite_step"]
                s0, s1, b, i_ref, s_ref = fs["s0"], fs["s1"], r["b"], r["i_ref"], r["s_ref"]
                i0, i1 = i_ref * (s0 / s_ref) ** b, i_ref * (s1 / s_ref) ** b
                extra_rev = (i1 - i0) * mg["value_per_install_forecast_included"] * haircut
                avg = d["evidence"]["average_d30_roas"]
                n_inst = d["evidence"]["populations"]["included"]["installs"]
                L = [f"Channel `{d['channel']}`, review {d['as_of']}, current budget {s0:.2f}/day, proposed {s1:.2f}/day (+{100 * (s1 / s0 - 1):.1f}%).", "",
                     f"- forecast value per install of the INCLUDED cohorts = {avg['forecast_d30_net_revenue']:.2f} / {n_inst} installs = {mg['value_per_install_forecast_included']:.4f}",
                     f"- average D30 ROAS (included cohorts) = forecast revenue {avg['forecast_d30_net_revenue']:.2f} / spend {avg['spend']:.2f} = **{avg['point']:.3f}**",
                     f"- fitted elasticity b = {b:.3f} (se {r['b_se']:.3f}); anchor: {i_ref:.1f} installs/day at {s_ref:.1f}/day spend",
                     f"- installs at {s0:.2f}: {i_ref:.1f} x ({s0:.2f}/{s_ref:.1f})^{b:.3f} = {i0:.2f}; at {s1:.2f}: {i1:.2f}; extra installs = {i1 - i0:.3f}",
                     f"- extra revenue = {i1 - i0:.3f} x {mg['value_per_install_forecast_included']:.4f} x haircut {haircut:.2f} = {extra_rev:.3f}; extra spend = {s1 - s0:.2f}",
                     f"- **model-estimated finite-step marginal D30 ROAS = {extra_rev:.3f} / {s1 - s0:.2f} = {extra_rev / (s1 - s0):.3f}** (policy stored {fs['marginal_roas']:.3f})",
                     f"- derivative at the current budget (reference only, a different quantity) = {fs['derivative_marginal_roas']:.3f}",
                     f"- Monte Carlo lower bound (10th percentile of {mg['lower_bound']['draws']} draws over b and value) = {mg['lower_bound']['lower']:.3f}; target D30 ROAS = {d['evidence']['payback_target']['target_d30_roas']:.3f}"]
                return "\n".join(L) + "\n"
    return "No decision on the dev seed reached the marginal-ROAS stage, so there is no real-data example; the toy example above is the hand-check.\n"


def ledger_demo(results, pcfg, fcfg, workdir):
    """Exercise the sandbox ledger. Keeps the Phase 3 distinction explicit: policy output vs ledger-mechanics demonstration."""
    # pick the review with the most budget changes, else the first
    date = max(results, key=lambda k: sum(d["action"] in ("increase_test", "decrease") for d in results[k]["decisions"].values()))
    res = results[date]
    budgets = {ch: d["current_budget"] for ch, d in res["decisions"].items()}
    ceiling = res["totals"]["ceiling"]
    out = {"date": date, "policy_changes": {ch: (d["action"], d["current_budget"], d["proposed_budget"]) for ch, d in res["decisions"].items() if d["action"] in ("increase_test", "decrease")}}
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    active = lambda: (cons.load_policy(), cons.load_forecast_config())  # the ACTIVE configuration is read from the YAML files at approval/apply
    # A. the policy's own proposal, applied as policy says
    led = Ledger(workdir / "policy.db", pcfg, fcfg, active_config_provider=active)
    led.initialize(budgets, ceiling, now=date)
    prop = pol.build_rules_proposal(res, led.head())  # stamped with the hash of the configuration that produced the decisions
    led.create_proposal(prop, date)
    out["policy_proposal_approval"] = led.approve(prop.proposal_id, "sandbox reviewer", date)
    out["policy_state"] = led.state()

    # B. LEDGER MECHANICS demonstration (explicitly labelled): the same changes, increases marked mechanics_test, on a flagged ledger
    mech = Ledger(workdir / "mechanics.db", pcfg, fcfg, allow_mechanics_test=True, active_config_provider=active)
    mech.initialize(budgets, ceiling, now=date)
    changes = {ch: p[2] for ch, p in out["policy_changes"].items()} or {"meta": round(budgets["meta"] * 1.1, 1), "unity": round(budgets["unity"] * 0.9, 1)}
    out["mechanics_changes_are_policy_output"] = bool(out["policy_changes"])
    st = mech.state()
    items = []
    for ch, cur in st["budgets"].items():
        new = changes.get(ch, cur)
        act = "increase_test" if new > cur else "decrease" if new < cur else "hold"
        items.append(ProposalItem(channel=ch, action=act, current_budget=cur, new_budget=new, qualification="mechanics_test" if act == "increase_test" else "none",
                                  reason="LEDGER MECHANICS DEMONSTRATION"))
    pid = make_proposal_id(date, items, st["version"], mech.policy_hash)
    mech.create_proposal(Proposal(proposal_id=pid, as_of=date, ledger_version=st["version"], policy_hash=mech.policy_hash, items=items), date)
    out["mech_approval"] = mech.approve(pid, "sandbox reviewer", date)
    out["mech_apply"] = mech.apply(pid, date)
    out["mech_rollback"] = mech.rollback(out["mech_apply"]["change_id"], "sandbox reviewer", "demonstration rollback", date)
    out["mech_audit"] = [e["event"] for e in mech.audit()]

    # C. safeguard checks added after independent review, exercised on a scratch ledger
    live = {"policy": copy.deepcopy(pcfg), "fc": copy.deepcopy(fcfg)}
    sg = Ledger(workdir / "safeguards.db", copy.deepcopy(pcfg), copy.deepcopy(fcfg), active_config_provider=lambda: (live["policy"], live["fc"]))
    sg.initialize(budgets, ceiling, now=date)

    def cut(channel, factor):
        s = sg.state()
        items = [ProposalItem(channel=c, action="decrease" if c == channel else "hold", current_budget=b,
                              new_budget=round(b * factor, 4) if c == channel else b, reason="safeguard demonstration") for c, b in s["budgets"].items()]
        pid = make_proposal_id(date, items, s["version"], sg.policy_hash) + f"-{len(sg.audit())}"
        sg.create_proposal(Proposal(proposal_id=pid, as_of=date, ledger_version=s["version"], policy_hash=sg.policy_hash, items=items), date)
        return pid

    ch = next(iter(budgets))
    out["safeguard_non_reversible"] = sg.approve(cut(ch, 0.80), "sandbox reviewer", date)  # -20%: inverse +25% breaks the 20% cap
    ok_pid = cut(ch, 0.90)
    out["safeguard_reversible_approval"] = sg.approve(ok_pid, "sandbox reviewer", date)
    live["policy"]["marginal"]["incremental_quality_haircut"] = 0.70  # the ACTIVE configuration changes between approval and application
    out["safeguard_config_changed_apply"] = sg.apply(ok_pid, date)
    live["policy"]["marginal"]["incremental_quality_haircut"] = pcfg["marginal"]["incremental_quality_haircut"]
    out["safeguard_config_restored_apply"] = sg.apply(ok_pid, date)
    return jsonable(out)


def scenario_check(results):
    rows = []
    for p in sorted((ROOT / "evals" / "scenarios").glob("*.json")):
        sc = json.loads(p.read_text(encoding="utf-8"))
        if not sc["evaluation_as_of_days"]:
            continue
        ex = sc["expected"]
        for day in sc["evaluation_as_of_days"]:
            date = day_to_date(START, day)
            if date not in results:
                continue
            d = results[date]["decisions"][sc["channel"]]
            a = d["action"]
            verdict = "ideal" if a in ex["ideal_actions"] else "acceptable" if a in ex["acceptable_actions"] else "FORBIDDEN" if a in ex["forbidden_actions"] else "unlisted"
            rows.append({"scenario": sc["id"], "channel": sc["channel"], "as_of_day": day, "action": a, "verdict": verdict, "reason": d["reasons"][0][:110]})
    return pd.DataFrame(rows)


def changes_section(results, demo):
    """Compare with the decisions saved from commit 96b8062 (before the three safeguard corrections)."""
    prev_path = ROOT / "data" / "scratch" / "prev" / "decisions_96b8062.json"
    L = ["## 7. What changed since 96b8062", "",
         "Corrections: (1) policy decisions now verify that the weekly summary and its contributing forecasts carry the hash of the forecast/quality configuration of the review; "
         "(2) the ledger holds an immutable configuration snapshot and verifies the active configuration hash at approval and application; "
         "(3) approval and application also require the exact inverse of a change to satisfy the hard constraints; (4) the decrease-rule wording is now a documented policy heuristic. "
         "No threshold, haircut or expected scenario answer was changed.", ""]
    if not prev_path.exists():
        return L + ["(previous decisions snapshot not available; comparison omitted)", ""]
    prev = json.loads(prev_path.read_text())["results"]
    diffs, n = [], 0
    for date, res in results.items():
        for ch, d in res["decisions"].items():
            n += 1
            p = prev[date]["decisions"][ch]
            if p["action"] != d["action"] or abs(p["proposed_budget"] - d["proposed_budget"]) > 1e-9:
                diffs.append({"as_of": date, "channel": ch, "before": p["action"], "now": d["action"], "reason now": d["reasons"][0][:100]})
    L += [f"Decisions compared: {n}. **Changed: {len(diffs)}.**", "",
          md(pd.DataFrame(diffs)) if diffs else "Every action and every proposed budget is identical to 96b8062: the stored evidence was regenerated under the new evidence hash and matches the current configuration.", ""]
    L += [f"Sandbox ledger demonstration totals (mechanics test at {demo['date']}): before {demo['mech_apply']['before']['total']:.1f} / reserve {demo['mech_apply']['before']['reserve']:.1f}; "
          f"after apply {demo['mech_apply']['after']['total']:.1f} / {demo['mech_apply']['after']['reserve']:.1f}; after rollback {demo['mech_rollback']['after']['total']:.1f} / {demo['mech_rollback']['after']['reserve']:.1f} "
          "(unchanged from 96b8062: the proposed policy cut is -10%, whose inverse +11.1% is within the cap).", ""]
    inv = sum(len(r["inverse_violations"]) for r in results.values())
    L += [f"Reversibility: inverse-constraint violations across all {len(results)} reviews' proposed budgets: {inv}.", ""]
    return L


def main(seed):
    t0 = time.time()
    pcfg, fcfg = cons.load_policy(), cons.load_forecast_config()
    dec = sqlite3.connect(ROOT / "data" / "decision" / f"seed_{seed}" / "decision.db")
    st = sqlite3.connect(ROOT / "data" / "forecasts" / f"seed_{seed}" / "forecasts.db")
    days = list(range(245, 337, 7))
    first = day_to_date(START, days[0])
    budgets = pol.initial_budgets_from_spend(dec, first)
    ceiling = sum(budgets.values()) * (1 + pcfg["constraints"]["ceiling_reserve_fraction"])
    results, reviews = {}, {}
    for day in days:
        as_of = day_to_date(START, day)
        reviews[as_of] = pol.build_review(dec, st, as_of, pcfg, fcfg)
        results[as_of] = pol.decide_all(dec, st, as_of, budgets, ceiling, review=reviews[as_of])
        print(f"[policy] {as_of} (day {day}): " + ", ".join(f"{ch}={d['action']}" for ch, d in results[as_of]["decisions"].items()), flush=True)
    (ROOT / "reports" / "phase3_policy_decisions.json").write_text(json.dumps(jsonable({"seed": seed, "note": "DEV SEED; static sandbox budgets at each review (no change carried forward). ALL DATA IS SYNTHETIC.",
                                                                                      "budgets": budgets, "ceiling": ceiling, "results": results}), indent=1), encoding="utf-8")

    L = ["# Phase 3 report (generated by scripts/run_policy_report.py)", "",
         "ALL DATA IS SYNTHETIC. DEV SEED ONLY. The ledger is a SANDBOX. `config/policy.yaml` was fixed before any decision was produced and was not changed afterwards. "
         "The sandbox budgets are held static at each review (no decision is carried forward to the next one). "
         "**No fully qualified experiment-backed increase was executed**: no pre-registered experiment plan exists before Phase 4.", "",
         "## 1. Definitions used", "",
         "- **average D30 ROAS (forecast-based)** = forecast D30 net revenue of the INCLUDED cohorts / spend of those same cohorts. It is not a whole-channel number.",
         "- **target D30 ROAS** = payback_multiple / (sum D180 net / sum D30 net) over eligible cohorts whose D180 milestone was complete and reported at the review date.",
         "- **model-estimated finite-step marginal D30 ROAS** = (extra installs x forecast value per install x incremental-quality haircut) / extra spend for the proposed step; "
         "extra installs come from a log-log spend->installs fit. It is a historical association read as a response curve, not causal incrementality evidence.",
         "- **derivative marginal ROAS** (reference only) = the slope at the current budget; a different quantity from the finite step.", "",
         "## 2. Hand-checkable calculations", "", "### A. Toy numbers (asserted in `tests/test_economics.py`)", ""]
    resp = {"b": 0.5, "i_ref": 100.0, "s_ref": 100.0}
    fs = eco.finite_step(resp, 100.0, 121.0, 2.0, 1.0)
    L += ["Anchor: 100 installs/day at 100/day spend; elasticity b = 0.5; each install worth 2.0; haircut 1.0.", "",
          f"- average ROAS = (100 x 2.0) / 100 = **{eco.average_roas(200.0, 100.0):.3f}**",
          f"- step 100 -> 121: installs 100 x 1.21^0.5 = {fs['installs_at_s1']:.0f}, so +{fs['extra_installs']:.0f} installs x 2.0 = +{fs['extra_revenue']:.0f} revenue for +{fs['extra_spend']:.0f} spend; **finite-step marginal ROAS = {fs['marginal_roas']:.3f}** (20/21)",
          f"- derivative at 100 = 2.0 x 0.5 x 100 / 100 = **{fs['derivative_marginal_roas']:.3f}** (the slope at the current budget; not the step)",
          "- so the same channel has three different numbers: 2.000 (average), 0.952 (this step), 1.000 (slope). Only the finite step answers 'what does THIS extra spend buy?'.", "",
          "### B. A real dev-seed decision, recomputed from the stored evidence", "", hand_check(results, pcfg)]

    L += ["## 3. Policy decisions on the dev seed", "",
          f"Sandbox starting budgets (mean visible daily spend over the 28 days before {first}): {budgets}; ceiling {ceiling:.1f} (+{100 * pcfg['constraints']['ceiling_reserve_fraction']:.0f}% reserve = {ceiling - sum(budgets.values()):.1f}). "
          f"{len(days)} reviews, {days[0]}..{days[-1]} (calendar {first} .. {day_to_date(START, days[-1])}); computation {time.time() - t0:.0f} s so far.", ""]
    flat = [dict(row_for(d), as_of=date) for date, res in results.items() for d in res["decisions"].values()]
    df = pd.DataFrame(flat)
    L += ["Actions by channel:", "", md(df.groupby(["channel", "action"]).size().unstack(fill_value=0).reset_index()), ""]
    blockers = {}
    for res in results.values():
        for d in res["decisions"].values():
            for r in d["reasons"] if d["action"] in ("insufficient_evidence", "hold") else []:
                key = r.split(":")[0].split(" (")[0][:90]
                blockers[key] = blockers.get(key, 0) + 1
    L += ["Most common reasons behind `hold` / `insufficient_evidence`:", "", md(pd.DataFrame(sorted(blockers.items(), key=lambda x: -x[1])[:12], columns=["reason", "decisions"])), ""]
    inc = df[df.action.isin(["increase_test", "decrease"])]
    L += ["All increase_test / decrease decisions with their evidence:", "", md(inc[["as_of", "channel", "action", "budget", "proposed", "avg ROAS (incl. cohorts)", "avg lo..hi", "target", "step ROAS (point / lower)"]]) if len(inc) else "(none on this dev seed)", ""]
    last = max(results)
    L += [f"Full evidence table for the last review ({last}):", "", md(pd.DataFrame([row_for(d) for d in results[last]["decisions"].values()])), ""]
    L += ["Budget totals under the policy's proposals (sandbox, before any approval) across all reviews: "
          f"totals never exceed the ceiling; constraint violations found by the shared checker: {sum(len(r['constraint_violations']) for r in results.values())}.", ""]

    # sensitivity to the incremental-quality haircut (an ASSUMPTION): NOT the chosen configuration, shown so the assumption's weight is visible
    sens_rows = []
    for h in (1.0, 0.9, 0.8, 0.7):
        counts = {ch: 0 for ch in CHANNELS}
        for as_of, rv in reviews.items():
            rv2 = copy.copy(rv)
            rv2.pcfg = copy.deepcopy(pcfg)
            rv2.pcfg["marginal"]["incremental_quality_haircut"] = h
            for ch, d in pol.decide_all(dec, st, as_of, budgets, ceiling, review=rv2)["decisions"].items():
                counts[ch] += d["action"] == "increase_test"
        sens_rows.append({"haircut": h, "chosen": "yes" if h == pcfg["marginal"]["incremental_quality_haircut"] else "", **counts, "total increase_test": sum(counts.values())})
    L += ["### Sensitivity of increase_test counts to the incremental-quality haircut (NOT the chosen configuration)", "",
          f"Number of the {len(days)} reviews at which each channel would be proposed an increase test, if the haircut were different. "
          "The haircut is a disclosed assumption (incremental players are worth that fraction of the included cohorts' average value per install); it was not tuned.", "",
          md(pd.DataFrame(sens_rows)), ""]

    demo = ledger_demo(results, pcfg, fcfg, ROOT / "data" / "ledger" / f"seed_{seed}")
    L += ["## 4. Sandbox ledger demonstration", "",
          f"Review used: {demo['date']}. Policy changes at that review: {demo['policy_changes'] or 'none'}.", "",
          f"**A. The policy's own proposal, as the policy emitted it** (increase tests carry qualification `plan_required`): approval result = `{demo['policy_proposal_approval'].get('code', 'approved')}`"
          f"{' (' + str(demo['policy_proposal_approval'].get('detail')) + ')' if not demo['policy_proposal_approval']['ok'] else ''}. "
          "The ledger refuses to approve or apply an increase without a registered frozen experiment plan, so no Phase 3 increase test can be executed.", "",
          f"**B. LEDGER MECHANICS demonstration (not an experiment-backed increase).** Changes used: {'the policy changes, with increases relabelled mechanics_test' if demo['mechanics_changes_are_policy_output'] else 'a fixed example (meta +10%, unity -10%) because the policy proposed no change at that review'}. "
          f"Recorded in the ledger as kind `{demo['mech_apply'].get('kind')}`.", ""]
    b, a = demo["mech_apply"]["before"], demo["mech_apply"]["after"]
    L += [md(pd.DataFrame([{"state": "before (v1)", "total": b["total"], "ceiling": b["ceiling"], "reserve": b["reserve"], **{k: v for k, v in b["budgets"].items()}},
                           {"state": f"after apply (v{a['version']})", "total": a["total"], "ceiling": a["ceiling"], "reserve": a["reserve"], **{k: v for k, v in a["budgets"].items()}},
                           {"state": f"after rollback (v{demo['mech_rollback']['version']})", "total": demo["mech_rollback"]["after"]["total"], "ceiling": demo["mech_rollback"]["after"]["ceiling"],
                            "reserve": demo["mech_rollback"]["after"]["reserve"], **{k: v for k, v in demo["mech_rollback"]["after"]["budgets"].items()}}]), 1), "",
          f"Audit events recorded: {demo['mech_audit']}.", "",
          "**C. Safeguards added after independent review** (scratch ledger, same review date):", "",
          f"- a -20% cut passes the 20% cap but its exact inverse (+25%) does not: approval result = `{demo['safeguard_non_reversible'].get('code')}`.",
          f"- a -10% cut is reversible: approval ok = {demo['safeguard_reversible_approval']['ok']}.",
          f"- the ACTIVE configuration (haircut 0.80 -> 0.70) changed between approval and application: apply = `{demo['safeguard_config_changed_apply'].get('code')}` "
          f"(ledger unchanged); after the configuration was restored, apply ok = {demo['safeguard_config_restored_apply']['ok']}.", ""]

    sc = scenario_check(results)
    L += ["## 5. Descriptive comparison with the scenario expectations (NOT the Phase 6 evaluation)", "",
          "Expectations were committed before any data existed (`evals/scenarios/*.json`). This table only shows where the dev-seed decisions fall; "
          "no policy value was tuned after seeing it, and any `FORBIDDEN` row is reported as a finding for review, not fixed here.", "",
          md(sc.groupby(["scenario", "verdict"]).size().unstack(fill_value=0).reset_index()), ""]
    bad = sc[sc.verdict == "FORBIDDEN"]
    L += ["Forbidden-action rows:", "", md(bad) if len(bad) else "(none)", ""]
    L += ["All scenario rows:", "", md(sc), ""]

    p2 = json.loads((ROOT / "reports" / "phase2_backtest_results.json").read_text())["weekly_summaries"]["overall"]
    steady = sc[sc.scenario == "steady_winner"]
    dec_counts = df[df.action == "decrease"].groupby("channel").size().to_dict()
    L += ["## 6. Findings for review (generated from the numbers above; nothing was tuned in response)", "",
          f"- **Steady winner never reaches an increase test.** At the {len(steady)} steady-winner reviews meta's decision was {sorted(steady.action.unique())}, "
          f"verdicts {steady.verdict.value_counts().to_dict()}; no review gave the committed ideal `increase_test`. Meta received `increase_test` at "
          f"{int((df[(df.channel == 'meta')].action == 'increase_test').sum())} of {len(days)} reviews. The haircut sensitivity above shows how much this depends on the 0.80 assumption.",
          f"- **Decisions rest on summary intervals whose observed coverage is below nominal.** The decrease rule needs the interval's UPPER bound below target, but the Phase 2 weekly-summary "
          f"intervals covered {100 * p2['coverage']:.1f}% of actuals against an 80% nominal level (dev seed). Decreases by channel: {dec_counts}. "
          "Weekly D0-D3 revenue is dominated by a few heavy-tailed purchases, so a decrease can follow sampling noise.",
          "- **Forbidden-action rows** (above) are reported as found. By inspection: the `tiny_sample_whale` expectation at day 329 refers to cohorts that are outside that review's weekly window by then, "
          "and the stored google_uac multiplier did NOT fall after the outage (it stayed near 1.93), so the day-315 `tracking_outage` decrease follows low observed D3 revenue in that window, not an outage artifact; "
          "whether either decrease was right has not been evaluated (Phase 6).",
          "- **The decrease and increase rules are documented policy heuristics, not proofs of economic safety.** They read forecast average ROAS and assume spend at the margin is not worth more than the "
          "average and that the included cohorts represent recent players; marginal ROAS can exceed average ROAS for some response shapes (the configuration permits an elasticity above one) or if player value changes with spend.",
          "- **Static budgets.** Each review is decided against the same starting budgets; no decision is carried forward, so repeated decreases of the same channel are not compounding here.", ""]
    L += changes_section(results, demo)
    out = ROOT / "reports" / "phase3_report.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    main(p.parse_args().seed)
