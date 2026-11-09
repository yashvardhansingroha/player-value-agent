"""Phase 4: one complete SIMULATED experiment lifecycle for the dev-seed policy increase test. ALL DATA IS SYNTHETIC.

    python scripts/run_experiment_lifecycle.py [--seed 1]

Needs data/decision/seed_<n>/decision.db, data/forecasts/seed_<n>/forecasts.db and data/hidden/seed_<n>/hidden_truth.json.
Writes reports/phase4_report.md and reports/phase4_results.json. Every number in the report comes from this run.

What it does (all labelled SIMULATED):
  1. recomputes the policy decision for the review (the dev seed's only increase test) with the frozen Phase 3 policy;
  2. builds the experiment plan for the dev-seed channel as it is (scale 1), runs the planning power study, and applies the declared
     adequacy standards; an inadequate design turns the action into insufficient_evidence and the real ledger refuses the plan;
  3. runs the declared scale search (what channel volume would this test need?);
  4. runs the full lifecycle at the first adequate scale: plan -> approval -> application -> pending -> readout -> keep/rollback;
  5. recomputes the readout independently, scores it against the simulator's truth AFTER the decision, and runs the evaluation-side replicate study.
No seed, threshold or standard was changed after seeing a result.
"""
import argparse
import copy
import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.metrics import milestone_complete_at  # noqa: E402
from src.experiments import design as D  # noqa: E402
from src.experiments import lifecycle as L  # noqa: E402
from src.experiments import plan as P  # noqa: E402
from src.experiments import readout as R  # noqa: E402
from src.experiments import store as E  # noqa: E402
from src.experiments.scoring import replicate_study, score_readout  # noqa: E402
from src.experiments.simulator import ExperimentSimulator, load_hidden, load_sim_config, value_reference  # noqa: E402
from src.ledger.ledger import Ledger  # noqa: E402
from src.ledger.proposal import Proposal, ProposalItem, jsonable  # noqa: E402
from src.policy import constraints as cons  # noqa: E402
from src.policy import policy as pol  # noqa: E402

AS_OF = "2025-10-13"
CHANNEL = "applovin"
OPERATOR = "sandbox reviewer"


def md(df, floats=3):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:.{floats}f}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(lines)


def build_world(truth, dec, decision, cfg, sim_cfg, scale, lag, baseline, proposed, pol_hash):
    """Units, simulator, pre-period observations and the plan for one scale (scale 1 = the dev-seed channel as it is)."""
    base = cfg["design"]["assignment_seed_base"]
    b = baseline * scale
    units = D.make_units(CHANNEL, b, cfg["design"]["units_per_channel"], D.derive_seed(base, "units", CHANNEL, AS_OF))
    v_ref = value_reference(dec, CHANNEL, AS_OF, sim_cfg["value_reference"]["matured_cohorts"])
    sim = ExperimentSimulator(truth, CHANNEL, {u["unit_id"]: u["baseline_daily_budget"] for u in units}, v_ref, sim_cfg, scale=scale,
                              unit_seed=D.derive_seed(base, "unit-truth", CHANNEL, AS_OF))
    obs_seed = D.derive_seed(base, "observations", CHANNEL, AS_OF)
    win = D.windows(AS_OF, lag, cfg["design"]["pre_days"], cfg["design"]["treatment_day_options"][0])
    pre_rows = sim.pre_rows(win, obs_seed)
    ev = decision["evidence"]
    target = {"target_d30_roas": ev["payback_target"]["target_d30_roas"],
              "source": {k: ev["payback_target"].get(k) for k in ("level", "n_cohorts", "installs", "first_install", "last_install", "d180_over_d30")}}
    fs, lb = ev["marginal_d30_roas_step"]["finite_step"], ev["marginal_d30_roas_step"]["lower_bound"]
    model_estimate = {"finite_step_marginal_roas": fs["marginal_roas"], "lower_bound": lb["lower"], "label": "MODEL-ESTIMATED (policy evidence)"}
    plan = D.build_plan(channel=CHANNEL, decision_as_of=AS_OF, policy_hash=pol_hash, baseline_budget=b, proposed_budget=proposed * scale, target=target,
                        model_estimate=model_estimate, units=units, pre_obs=pd.DataFrame(pre_rows), lag_days=lag, cfg=cfg,
                        scale_note=None if scale == 1 else f"SIMULATED SCALE-UP x{scale}: budgets and install volume multiplied; not the dev-seed channel as it is")
    return {"units": units, "sim": sim, "pre_rows": pre_rows, "plan": plan, "obs_seed": obs_seed, "scale": scale, "v_ref": v_ref}


def power_table(plan):
    rows = []
    for dur, blk in plan["power_assessment"]["per_duration"].items():
        for a in blk["alternatives"]:
            rows.append({"treatment days": int(dur), "true iROAS / target": a["multiple"], "true iROAS": a["true_iroas"], "P(keep)": a["keep_rate"], "MC SE": a["mc_se"],
                         "median estimate": a["median_estimate"], "median lower bound": a["median_lower_bound"], "replicates": blk["replicates"],
                         "role": ("POWER (declared alternative)" if abs(a["multiple"] - plan["power_assessment"]["standards"]["alternative_multiple_of_target"]) < 1e-9 else
                                 "FALSE-KEEP (boundary/below)" if a["multiple"] <= plan["power_assessment"]["standards"]["boundary_multiple"] + 1e-9 else "informational")})
    return pd.DataFrame(rows)


def independent_readout(obs: pd.DataFrame, plan: dict) -> dict:
    """Recompute the estimate with plain pandas (no design.py), to show the readout is not just trusting its own code."""
    w, treated = plan["windows"], {u["unit_id"] for u in plan["units"] if u["arm"] == "treatment"}
    pre = obs[(obs.phase == "pre") & obs.acquisition_date.between(w["pre_start"], w["pre_end"])].groupby("unit_id")[["spend", "d30_net_revenue"]].mean()
    post = obs[(obs.phase == "treatment")].groupby("unit_id")[["spend", "d30_net_revenue"]].mean()
    u = pd.DataFrame({"dR": post.d30_net_revenue - pre.d30_net_revenue, "dS": post.spend - pre.spend})
    u["arm"] = ["treatment" if i in treated else "control" for i in u.index]
    T, C = u[u.arm == "treatment"], u[u.arm == "control"]
    N, Dd = T.dR.mean() - C.dR.mean(), T.dS.mean() - C.dS.mean()
    r = N / Dd
    e = u.dR - r * u.dS
    se = np.sqrt(e[T.index].var(ddof=1) / len(T) + e[C.index].var(ddof=1) / len(C)) / abs(Dd)
    tcrit = sps.t.ppf(1 - plan["estimator"]["alpha"], len(T) + len(C) - 2)
    return {"units": u.assign(residual=e), "N": N, "D": Dd, "iroas": r, "se": se, "t_crit": tcrit, "lower": r - tcrit * se, "n_T": len(T), "n_C": len(C)}


def main(seed):
    t0 = time.time()
    cfg, sim_cfg, pcfg, fcfg = D.load_config(), load_sim_config(), cons.load_policy(), cons.load_forecast_config()
    truth = load_hidden(seed)
    dec = sqlite3.connect(ROOT / "data" / "decision" / f"seed_{seed}" / "decision.db")
    st = sqlite3.connect(ROOT / "data" / "forecasts" / f"seed_{seed}" / "forecasts.db")
    out_dir = ROOT / "data" / "experiments" / f"seed_{seed}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    R_ = {"label": "SIMULATED. ALL DATA IS SYNTHETIC.", "seed": seed, "as_of": AS_OF, "channel": CHANNEL}

    # ---- 1. the policy decision being tested
    budgets = pol.initial_budgets_from_spend(dec, "2025-09-08")  # same sandbox starting budgets as the Phase 3 report
    ceiling = sum(budgets.values()) * (1 + pcfg["constraints"]["ceiling_reserve_fraction"])
    result = pol.decide_all(dec, st, AS_OF, budgets, ceiling, pcfg, fcfg)
    decision = result["decisions"][CHANNEL]
    assert decision["action"] == "increase_test", f"the dev-seed review no longer proposes an increase test for {CHANNEL}: {decision['action']}"
    baseline, proposed = decision["current_budget"], decision["proposed_budget"]
    comp3 = milestone_complete_at(dec, AS_OF, 3)
    c = comp3[comp3.channel == CHANNEL]
    lag = int(max((pd.Timestamp(a) - pd.Timestamp(b)).days - 3 for a, b in zip(c.complete_at, c.install_date)))
    R_["decision"] = {"baseline": baseline, "proposed": proposed, "step": round(proposed - baseline, 6), "policy_hash": result["policy_hash"], "declared_reporting_lag_days": lag,
                      "target_d30_roas": decision["evidence"]["payback_target"]["target_d30_roas"], "average_d30_roas": decision["evidence"]["average_d30_roas"]["point"],
                      "model_estimated_finite_step_marginal_roas": decision["evidence"]["marginal_d30_roas_step"]["finite_step"]["marginal_roas"],
                      "lower_bound": decision["evidence"]["marginal_d30_roas_step"]["lower_bound"]["lower"], "reasons": decision["reasons"]}
    print(f"[1] decision {CHANNEL} {AS_OF}: increase_test {baseline} -> {proposed}", flush=True)

    # ---- 2. the design for the channel as it is (scale 1)
    w1 = build_world(truth, dec, decision, cfg, sim_cfg, 1, lag, baseline, proposed, result["policy_hash"])
    plan1 = w1["plan"]
    after = L.decision_after_design(decision, plan1)
    R_["scale1"] = {"plan_hash": P.plan_hash(plan1), "adequate": plan1["power_assessment"]["adequate"], "reasons": plan1["power_assessment"]["reasons"], "action_after_design": after["action"],
                    "power_table": power_table(plan1).to_dict("records")}
    led1 = Ledger(out_dir / "ledger_scale1.db", pcfg, fcfg, active_config_provider=lambda: (cons.load_policy(), cons.load_forecast_config()))
    led1.initialize({k: v for k, v in budgets.items()}, ceiling, now=AS_OF)
    led1.register_frozen_plan(plan1, "designer", AS_OF)
    src1 = pol.build_rules_proposal(result, led1.head())
    exp1 = L.experiment_proposal(src1, CHANNEL, plan1, led1.head())
    led1.create_proposal(exp1, AS_OF)
    R_["scale1"]["ledger_approval"] = jsonable(led1.approve(exp1.proposal_id, OPERATOR, AS_OF))
    print(f"[2] scale 1: adequate={plan1['power_assessment']['adequate']} -> action {after['action']}; ledger: {R_['scale1']['ledger_approval'].get('code', 'approved')}", flush=True)

    # ---- 3. declared scale searches: the dev-seed world (config/experiment.yaml), then the declared DEMONSTRATION world (config/experiment_demo.yaml)
    def search_scales(options, sim_cfg_x, label, reuse_first=None):
        rows, found = [], None
        for s in options:
            w = reuse_first if (s == 1 and reuse_first is not None) else build_world(truth, dec, decision, cfg, sim_cfg_x, s, lag, baseline, proposed, result["policy_hash"])
            pa = w["plan"]["power_assessment"]
            blk = pa["per_duration"][str(cfg["design"]["treatment_day_options"][-1])]
            std = pa["standards"]
            by = {a["multiple"]: a for a in blk["alternatives"]}
            rows.append({"world": label, "scale": s, "adequate": pa["adequate"], "chosen_treatment_days": pa["chosen_treatment_days"],
                         "power at 1.5x (56d)": by[std["alternative_multiple_of_target"]]["keep_rate"], "false-keep at 1.0x (56d)": by[std["boundary_multiple"]]["keep_rate"],
                         "median SE of iROAS (56d, at 1.5x)": by[std["alternative_multiple_of_target"]]["median_se"], "reasons": "; ".join(pa["reasons"])[:140]})
            print(f"[3] {label} scale {s}: adequate={pa['adequate']} days={pa['chosen_treatment_days']}", flush=True)
            if pa["adequate"] and found is None:
                found = w
                break
        return rows, found

    search, chosen = search_scales(cfg["scale_search"]["options"], sim_cfg, "dev-seed world", reuse_first=w1)
    R_["scale_search"] = search
    world_label = "dev-seed world"
    if chosen is None:  # the real result: inadequate everywhere. Now the pre-declared demonstration world (committed in 4b before this ran).
        with open(ROOT / "config" / "experiment_demo.yaml", encoding="utf-8") as f:
            import yaml
            demo = yaml.safe_load(f)
        sim_cfg_demo = copy.deepcopy(sim_cfg)
        for sect, vals in demo["sim_overrides"].items():
            sim_cfg_demo[sect].update(vals)
        demo_rows, chosen = search_scales(demo["scale_search"]["options"], sim_cfg_demo, "DEMONSTRATION world")
        R_["demo_search"] = demo_rows
        R_["demo_world"] = {"sim_overrides": demo["sim_overrides"], "scale_options": demo["scale_search"]["options"]}
        world_label = "DEMONSTRATION world"
        sim_cfg = sim_cfg_demo
        if chosen is None:
            R_["lifecycle"] = None
            write(R_, None, cfg, t0)
            print("no declared scale in either world made the design adequate; lifecycle not run", flush=True)
            return
    S, plan, sim = chosen["scale"], chosen["plan"], chosen["sim"]
    h = P.plan_hash(plan)
    R_["chosen_scale"], R_["chosen_world"] = S, world_label

    # ---- 4. the full lifecycle at that scale, through the real ledger
    scaled_budgets = {k: v * S for k, v in budgets.items()}
    led = Ledger(out_dir / "ledger_lifecycle.db", pcfg, fcfg, active_config_provider=lambda: (cons.load_policy(), cons.load_forecast_config()))
    led.initialize(scaled_budgets, ceiling * S, now=AS_OF)
    ec = E.create_store(out_dir / "experiment.db")
    E.set_clock(ec, AS_OF)
    E.add_units(ec, CHANNEL, chosen["units"])
    E.insert_observations(ec, chosen["pre_rows"])  # the pre-period: observed history available at the decision date
    hash_ledger = led.register_frozen_plan(plan, "designer", AS_OF)
    hash_store = E.register_plan(ec, plan, cfg)  # the store checks the plan against the authoritative experiment configuration too
    assert hash_ledger == hash_store == h
    src = pol.build_rules_proposal(result, led.head())
    scaled_items = [ProposalItem(channel=i.channel, action=i.action, current_budget=i.current_budget * S, new_budget=round(i.new_budget * S, 6), qualification=i.qualification,
                                 experiment_plan_hash=None, evidence=i.evidence, reason=i.reason) for i in src.items]
    scaled_src = Proposal(proposal_id="P-scaled-policy", as_of=AS_OF, ledger_version=led.head(), policy_hash=src.policy_hash, items=scaled_items)
    prop = L.experiment_proposal(scaled_src, CHANNEL, plan, led.head())
    led.create_proposal(prop, AS_OF)
    transitions = []

    def snap(label, extra=""):
        s = led.state()
        transitions.append({"step": label, "ledger version": s["version"], "kind": s["kind"], "total": s["total"], "ceiling": s["ceiling"], "reserve": s["reserve"],
                            CHANNEL: s["budgets"][CHANNEL], "sim clock": E.clock(ec), "note": extra})

    snap("initial ledger")
    snap("plan frozen (ledger + experiment store), proposal created", f"plan {h[:12]}...")
    approval = led.approve(prop.proposal_id, OPERATOR, AS_OF)
    assert approval["ok"], approval
    snap("approved (bound to proposal, ledger version, policy hash, plan hash)")
    applied = led.apply(prop.proposal_id, AS_OF)
    assert applied["ok"] and applied["kind"] == "APPLY", applied
    L.start_experiment(led, ec, h, applied["change_id"])  # verified against the ledger: the active experiment-backed APPLY of exactly this plan
    snap("applied: experiment-backed increase", f"kind {applied['kind']} (not MECHANICS_TEST)")
    exec_kw = {"policy_cfg": pcfg, "forecast_cfg": fcfg, "experiment_cfg": cfg}
    authoritative = []

    def fabricated(decision):
        """What a dishonest caller could hand in: the right plan hash and the decision they want."""
        return {"label": "SIMULATED READOUT. ALL DATA IS SYNTHETIC.", "plan_hash": h, "clock": E.clock(ec), "decision": decision,
                "decision_code": "keep" if decision == "keep" else "rollback:lower_bound_below_target",
                "estimate": {"iroas": 9.99, "se": 0.01, "lower_bound": 9.9}, "rule": {"lower_bound_meets_target": decision == "keep"}}

    for decision in ("keep", "rollback"):  # fabricated results while the experiment is still pending: refused, nothing moves
        head_before = led.head()
        r = L.execute_readout(led, ec, h, "caller", preview=fabricated(decision), **exec_kw)
        authoritative.append({"attempt": f"fabricated '{decision}' result at clock {E.clock(ec)} (before the readout date)", "response": r["code"],
                              "outcome recorded": r["outcome_recorded"], "ledger version unchanged": led.head() == head_before})
    w = plan["windows"]
    pending = []

    def try_readout(label):
        try:
            R.run_readout(plan, E.load_observations(ec), E.clock(ec), pcfg, fcfg, cfg)
            pending.append({"attempt": label, "result": "readout ran"})
        except R.ReadoutNotAvailable as e:
            pending.append({"attempt": label, "result": "REFUSED: " + str(e), "pending acquisition days": len(e.pending_days)})

    try_readout("on the application date")
    for label, days in (("treatment window mid-point", w["treatment_days"] // 2), ("end of treatment window", w["treatment_days"] - 1), ("day before the last cohort's D30 outcome + lag", None)):
        clock = P.add_days(w["treatment_start"], days) if days is not None else P.add_days(w["readout_available_on"], -1)
        rel = L.advance_clock(ec, sim, plan, clock, chosen["obs_seed"])
        try_readout(f"{label} (clock {clock}; {rel['total_treatment_rows']} of {len(plan['units']) * w['treatment_days']} outcome rows released)")
    snap("pending: clock advanced, readout still refused", f"{len(pending)} early attempts refused")
    L.advance_clock(ec, sim, plan, w["readout_available_on"], chosen["obs_seed"])
    obs = E.load_observations(ec)
    readout = R.run_readout(plan, obs, E.clock(ec), pcfg, fcfg, cfg)
    snap("readout available and computed", f"decision {readout['decision']} ({readout['decision_code']})")
    opposite = "keep" if readout["decision"] == "rollback" else "rollback"
    head_before = led.head()
    flipped = dict(readout, decision=opposite, decision_code="keep" if opposite == "keep" else "rollback:lower_bound_below_target")
    r = L.execute_readout(led, ec, h, "caller", preview=flipped, **exec_kw)
    authoritative.append({"attempt": f"valid readout with the decision edited to '{opposite}'", "response": r["code"], "outcome recorded": r["outcome_recorded"],
                          "ledger version unchanged": led.head() == head_before})
    exec_res = L.execute_readout(led, ec, h, "readout rule", preview=readout, **exec_kw)  # the preview matches the recomputed readout, which is what gets executed
    assert exec_res["readout"]["decision"] == readout["decision"]
    snap(f"readout executed: {exec_res['status']}")
    dup = L.execute_readout(led, ec, h, "readout rule", preview=readout, **exec_kw)
    R_["lifecycle"] = {"scale": S, "plan_hash": h, "approval": jsonable(approval), "apply": {"change_id": applied["change_id"], "kind": applied["kind"]}, "pending_attempts": pending,
                       "readout": readout, "execution": jsonable({k: v for k, v in exec_res.items() if k not in ("before", "after", "readout")}), "duplicate_execution": dup,
                       "authoritative_execution_checks": authoritative,
                       "transitions": transitions, "audit_events": [e["event"] for e in led.audit()],
                       "experiment_events": [r[0] for r in ec.execute("SELECT event FROM experiment_events ORDER BY id")]}

    # ---- rollback_blocked demonstration on a scratch copy of the same experiment (a later change touched the channel)
    led_b = Ledger(out_dir / "ledger_blocked.db", pcfg, fcfg)
    led_b.initialize(scaled_budgets, ceiling * S, now=AS_OF)
    led_b.register_frozen_plan(plan, "designer", AS_OF)
    pb = L.experiment_proposal(scaled_src, CHANNEL, plan, led_b.head())
    led_b.create_proposal(pb, AS_OF)
    led_b.approve(pb.proposal_id, OPERATOR, AS_OF)
    ab = led_b.apply(pb.proposal_id, AS_OF)
    sb = led_b.state()
    items = [ProposalItem(channel=k, action="decrease" if k == CHANNEL else "hold", current_budget=v, new_budget=round(v * 0.95, 6) if k == CHANNEL else v, reason="a later unrelated change")
             for k, v in sb["budgets"].items()]
    later = Proposal(proposal_id="P-later-change", as_of="2025-10-20", ledger_version=sb["version"], policy_hash=led_b.policy_hash, items=items)
    led_b.create_proposal(later, "2025-10-20")
    led_b.approve("P-later-change", OPERATOR, "2025-10-20")
    led_b.apply("P-later-change", "2025-10-20")
    ec_b = E.create_store(out_dir / "experiment_blocked.db")
    E.set_clock(ec_b, AS_OF)
    E.add_units(ec_b, CHANNEL, chosen["units"])
    E.insert_observations(ec_b, chosen["pre_rows"])
    E.register_plan(ec_b, plan, cfg)
    L.start_experiment(led_b, ec_b, h, ab["change_id"])
    L.advance_clock(ec_b, sim, plan, w["readout_available_on"], chosen["obs_seed"])
    rb = R.run_readout(plan, E.load_observations(ec_b), E.clock(ec_b), pcfg, fcfg, cfg)
    blocked = L.execute_readout(led_b, ec_b, h, "readout rule", preview=rb, **exec_kw)
    # ... and the configuration-not-loaded refusal
    other = copy.deepcopy(pcfg)
    other["marginal"]["incremental_quality_haircut"] = 0.70
    led_b.reload_config(other, fcfg, w["readout_available_on"])
    cfg_refused = L.execute_readout(led_b, ec_b, h, "readout rule", preview=rb, **exec_kw)
    R_["blocked_demo"] = {"readout_decision": rb["decision"], "rollback_attempt": jsonable(blocked), "configuration_not_loaded": jsonable(cfg_refused)}

    # ---- 5. independent recomputation, scoring against truth (after the decision), evaluation-side replicate study
    ind = independent_readout(obs, plan)
    R_["independent"] = {k: float(ind[k]) for k in ("N", "D", "iroas", "se", "t_crit", "lower")} | {"matches_readout": bool(
        abs(ind["iroas"] - readout["estimate"]["iroas"]) < 1e-9 and abs(ind["lower"] - readout["estimate"]["lower_bound"]) < 1e-9)}
    R_["score"] = score_readout(plan, readout, sim)
    n_rep = 300
    R_["replicate_study_chosen_design"] = replicate_study(plan, sim, n_rep, D.derive_seed(cfg["power"]["seed"], "evaluation"), reference_estimate=readout["estimate"]["iroas"])
    sim1 = w1["sim"]
    R_["replicate_study_scale1_design"] = replicate_study(plan1, sim1, n_rep, D.derive_seed(cfg["power"]["seed"], "evaluation-scale1"))
    R_["truth_scale1"] = sim1.true_effect(plan1)
    R_["elapsed_seconds"] = round(time.time() - t0, 1)
    write(R_, {"plan": plan, "plan1": plan1, "ind": ind, "obs": obs, "result": result, "decision": decision}, cfg, t0)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


def write(R_, extra, cfg, t0):
    (ROOT / "reports" / "phase4_results.json").write_text(json.dumps(jsonable(R_), indent=1), encoding="utf-8")
    if extra is None:
        (ROOT / "reports" / "phase4_report.md").write_text("# Phase 4 report\n\nNo declared scale made the design adequate; lifecycle not run.\n", encoding="utf-8")
        return
    plan, plan1, ind, obs = extra["plan"], extra["plan1"], extra["ind"], extra["obs"]
    life, d = R_["lifecycle"], R_["decision"]
    pa, std = plan["power_assessment"], plan["power_assessment"]["standards"]
    L_ = ["# Phase 4 report (generated by scripts/run_experiment_lifecycle.py)", "",
          "**SIMULATED. ALL DATA IS SYNTHETIC.** Every experiment here is a simulation on synthetic units; nothing is real incrementality evidence. "
          "The design standards (`config/experiment.yaml`) and simulator parameters (`config/experiment_sim.yaml`) were committed in `aecc836` before any experiment was simulated, and no seed, threshold or standard was changed after a result was seen.", "",
          "## 1. The decision being tested", "",
          f"The frozen Phase 3 policy proposes ONE increase test on the dev seed: `{R_['channel']}` at review {R_['as_of']}, daily budget {d['baseline']} -> {d['proposed']} (step {d['step']}). "
          f"Target D30 ROAS {d['target_d30_roas']:.3f}; forecast average D30 ROAS (included cohorts) {d['average_d30_roas']:.3f}; MODEL-ESTIMATED finite-step marginal D30 ROAS {d['model_estimated_finite_step_marginal_roas']:.3f} "
          f"with lower bound {d['lower_bound']:.3f}. The increase needs a pre-registered experiment before the ledger will touch it.", "",
          "## 2. Design for the channel as it is (scale 1) and its verdict", "",
          f"24 units partition the channel budget; 12 are treated and receive the whole step. Planning simulation: {cfg['power']['replicates']} replicates per cell, same estimator and rule as the readout, noise bootstrapped from the PRE-period observations only. "
          f"Standards: power >= {std['target_power']} at {std['alternative_multiple_of_target']}x target; false-keep <= {std['max_false_keep']} (+{std['false_keep_tolerance_mc_se']} MC SE) at {std['boundary_multiple']}x target and at {std['below_boundary_multiples']}x.", "",
          md(power_table(plan1).query("`treatment days` == 56 or `treatment days` == 14")), "",
          f"**Verdict at scale 1: {'ADEQUATE' if R_['scale1']['adequate'] else 'INADEQUATE'}** ({'; '.join(R_['scale1']['reasons'])}). "
          f"The policy action becomes `{R_['scale1']['action_after_design']}`. The real ledger refuses to approve this real plan: "
          f"`{R_['scale1']['ledger_approval'].get('code', 'approved')}`.", "",
          "## 3. What volume would this test need? (declared scale search; SIMULATED scale-up of budgets and installs)", "",
          md(pd.DataFrame(R_["scale_search"])), "",
          "**Real result: no scale in the declared list makes the design adequate for the dev-seed world.** The test itself is valid (false-keep at the target boundary stays near 0.10) but weak: "
          "heavy-tailed in-app purchase revenue averages out only as 1/sqrt(volume), and per-unit-day installs carry about 50% relative overdispersion that no extra volume removes, so the standard error of iROAS stays near 0.4 even at x100 volume.", "",
          "### The declared DEMONSTRATION world (config/experiment_demo.yaml, committed in `4d5c7f5` BEFORE it was run)", "",
          "Because no adequate plan exists for the dev-seed world, the full lifecycle (which needs an adequate plan to pass the ledger gate) is exercised in a separately declared synthetic world: far larger volume and near-Poisson count noise. "
          "The design standards, estimator, guardrails and decision rule are unchanged; only the world's noise and volume differ. It is NOT the dev-seed channel and says nothing about it. The keep/rollback outcome comes from the simulator's truth and was not chosen.", "",
          f"Simulator overrides: {R_['demo_world']['sim_overrides']}; scale options {R_['demo_world']['scale_options']}.", "",
          md(pd.DataFrame(R_["demo_search"])), "",
          f"The first adequate scale in the demonstration world is **x{R_['chosen_scale']}**. The lifecycle below runs there, with a real, adequate plan (no `mechanics_test`).", "",
          "**Disclosure:** the first run of this script had an integer-overflow bug in the exact budget allocation (the product step x budget in millionths exceeded int64 for budgets above about 1e9 millionths, i.e. scale x100 and up), "
          "which silently corrupted the treated increments in the planning study. It was found when a plan failed structural validation, fixed before the final run (regression test added), and every number in this report comes from the corrected code. "
          "The conclusion for the dev-seed world (inadequate at every declared scale) is unchanged; the demonstration world's first adequate scale moved from x10000 to x1000.", "",
          f"## 4. The frozen plan (demonstration world, scale x{R_['chosen_scale']})", "",
          f"- plan hash (SHA-256 of the canonical content, recomputed by the ledger and the experiment store): `{life['plan_hash']}`",
          f"- channel `{plan['channel']}`; decision date {plan['decision_as_of']}; policy hash `{plan['policy_hash'][:16]}...`; experiment-config hash `{plan['experiment_config_hash'][:16]}...`",
          f"- budget step: {plan['budget_step']['baseline_daily_budget']:.2f} -> {plan['budget_step']['proposed_daily_budget']:.2f} per day (step {plan['budget_step']['step_amount']:.2f}, {100 * plan['budget_step']['step_pct']:.1f}%)",
          f"- units: {len(plan['units'])} (treatment {sum(u['arm'] == 'treatment' for u in plan['units'])}); baseline budgets sum to {sum(u['baseline_daily_budget'] for u in plan['units']):.6f}; "
          f"treated increments sum to **{sum(u['planned_incremental_spend'] for u in plan['units'] if u['arm'] == 'treatment'):.6f}** = the exact approved step",
          f"- assignment: {plan['assignment']['method']} (seed {plan['assignment']['seed']})",
          f"- windows: pre {plan['windows']['pre_start']}..{plan['windows']['pre_end']} ({plan['windows']['pre_days']} days, outcomes available at the decision date); treatment {plan['windows']['treatment_start']}..{plan['windows']['treatment_end']} "
          f"({plan['windows']['treatment_days']} days, shortest adequate); D30 horizon {plan['windows']['outcome_horizon_days']} + reporting lag {plan['windows']['reporting_lag_days']} -> **readout available {plan['windows']['readout_available_on']}**",
          f"- estimator: unit-level difference-in-differences ratio; one-sided alpha {plan['estimator']['alpha']}; delivery must be {plan['estimator']['min_delivery_fraction']}..{plan['estimator']['max_delivery_fraction']} of plan",
          f"- guardrails: CPI ratio <= {plan['guardrails']['max_cpi_ratio']}; D1 retention change >= {plan['guardrails']['min_d1_retention_change']}",
          f"- rule: {plan['decision_rule']['keep_if']} (target {plan['decision_rule']['target_d30_roas']:.4f})", "",
          "Planning study at the chosen window:", "", md(power_table(plan).query(f"`treatment days` == {plan['windows']['treatment_days']}")), ""]
    # hand-checkable readout
    u = ind["units"].round(4).reset_index().rename(columns={"index": "unit_id", "dR": "revenue change/day", "dS": "spend change/day"})
    ro = life["readout"]
    L_ += ["## 5. The readout (SIMULATED), recomputed by hand", "",
           "Each unit: average daily D30 net revenue and spend AFTER the change minus BEFORE. Treated and control units are then compared.", "",
           md(u.head(6), 3) + "\n| ... | ... | ... | ... | ... |", "",
           f"- mean revenue change: treated {ind['units'][ind['units'].arm == 'treatment'].dR.mean():.3f}, control {ind['units'][ind['units'].arm == 'control'].dR.mean():.3f}  ->  N = {ind['N']:.4f} extra revenue per day (per unit)",
           f"- mean spend change: treated {ind['units'][ind['units'].arm == 'treatment'].dS.mean():.3f}, control {ind['units'][ind['units'].arm == 'control'].dS.mean():.3f}  ->  D = {ind['D']:.4f} extra spend per day (per unit)",
           f"- **iROAS = N / D = {ind['iroas']:.4f}**; residual e = dR - iROAS x dS; SE = sqrt(var_T/{ind['n_T']} + var_C/{ind['n_C']}) / |D| = {ind['se']:.4f}; one-sided t({1 - plan['estimator']['alpha']:.2f}, df {ind['n_T'] + ind['n_C'] - 2}) = {ind['t_crit']:.4f}",
           f"- **lower bound = {ind['iroas']:.4f} - {ind['t_crit']:.4f} x {ind['se']:.4f} = {ind['lower']:.4f}** versus the frozen target {ro['rule']['target_d30_roas']:.4f}",
           f"- independent pandas recomputation matches the readout: **{R_['independent']['matches_readout']}** (readout: iROAS {ro['estimate']['iroas']:.4f}, lower bound {ro['estimate']['lower_bound']:.4f})",
           f"- guardrails: CPI ratio {ro['guardrails']['cpi_ratio']:.3f} (limit {ro['guardrails']['cpi_ratio_limit']}); D1 retention change {ro['guardrails']['d1_retention_change']:+.4f} (floor {ro['guardrails']['d1_retention_change_floor']})",
           f"- **decision: {ro['decision'].upper()}** (`{ro['decision_code']}`)", "",
           "### Planned versus realised spending (reported separately)", "",
           md(pd.DataFrame([{"": "planned extra spend per day (all treated units)", "value": ro["spending"]["planned_extra_spend_per_day_total"]},
                            {"": "planned extra spend over the window", "value": ro["spending"]["planned_extra_spend_over_window"]},
                            {"": "realised extra spend per day (estimate from observed spend)", "value": ro["spending"]["realised_extra_spend_per_day_total_estimate"]},
                            {"": "realised extra spend over the window (estimate)", "value": ro["spending"]["realised_extra_spend_over_window_estimate"]},
                            {"": "delivery fraction (realised / planned)", "value": ro["spending"]["delivery_fraction"]}]), 4), "",
           "## 6. Ledger and clock transitions", "", md(pd.DataFrame(life["transitions"]), 2), "",
           "Early readout attempts (all must be refused until the last acquisition cohort's D30 outcome and reporting delay have elapsed):", "",
           md(pd.DataFrame(life["pending_attempts"])), "",
           f"Ledger audit events in order: {life['audit_events']}.", f"Experiment events: {life['experiment_events']}.", "",
           f"Duplicate execution of the same readout: `{life['duplicate_execution'].get('code')}`.", "",
           "**Authoritative execution.** `execute_readout` takes only the plan hash. It re-verifies the registered plan (not altered, valid, follows the ledger's experiment configuration, adequate), "
           "checks that the run link is the active experiment-backed APPLY of exactly this plan, reads the experiment store's own clock and observations, recomputes the readout with the production "
           "function under the explicitly supplied configuration, and executes only that. A caller-supplied readout is a preview that must match. Refusals recorded in this run:", "",
           md(pd.DataFrame(life["authoritative_execution_checks"])), "",
           f"**rollback_blocked demonstration** (scratch copy; a later unrelated change touched the channel before the readout said {R_['blocked_demo']['readout_decision']}): ledger status "
           f"`{R_['blocked_demo']['rollback_attempt'].get('status')}`, reason `{R_['blocked_demo']['rollback_attempt'].get('ledger', {}).get('code')}`, outcome recorded = {R_['blocked_demo']['rollback_attempt'].get('outcome_recorded')}. "
           f"**Configuration not loaded:** after the ledger was reloaded with a different configuration, executing the readout is refused with `{R_['blocked_demo']['configuration_not_loaded'].get('status')}`.", ""]
    sc, rs, rs1 = R_["score"], R_["replicate_study_chosen_design"], R_["replicate_study_scale1_design"]
    L_ += ["## 7. Statistical diagnostics (SIMULATED)", "",
           "### Scored against the simulator's truth AFTER the decision (hidden truth never supplied the observed result)", "",
           md(pd.DataFrame([{"quantity": k, "value": v} for k, v in sc.items() if isinstance(v, (int, float, bool))]), 4), "",
           f"### Evaluation-side replicate study: the actual simulator re-run {rs['replicates']} times through the SAME estimator and rule", "",
           md(pd.DataFrame([{"design": f"chosen (scale x{R_['chosen_scale']})", "true iROAS": rs["true_iroas_treated_units"], "target": rs["target_d30_roas"], "P(keep)": rs["keep_rate"], "MC SE": rs["keep_rate_mc_se"],
                             "estimate mean": rs["estimate_mean"], "bias": rs["estimate_bias"], "SD": rs["estimate_sd"], "lower-bound coverage of truth": rs["lower_bound_coverage_of_truth"], "nominal": rs["nominal_one_sided_coverage"]},
                            {"design": "dev-seed channel as it is (scale 1, inadequate)", "true iROAS": rs1["true_iroas_treated_units"], "target": rs1["target_d30_roas"], "P(keep)": rs1["keep_rate"], "MC SE": rs1["keep_rate_mc_se"],
                             "estimate mean": rs1["estimate_mean"], "bias": rs1["estimate_bias"], "SD": rs1["estimate_sd"], "lower-bound coverage of truth": rs1["lower_bound_coverage_of_truth"], "nominal": rs1["nominal_one_sided_coverage"]}])), "",
           f"Where the actual readout falls: its iROAS estimate {rs['reference_estimate']:.4f} sits at the **{100 * rs['reference_estimate_percentile']:.0f}th percentile** of the {rs['replicates']} replicate estimates "
           f"(5th/50th/95th: {', '.join(f'{x:.3f}' for x in rs['estimate_quantiles_5_50_95'])}); the database path and the in-memory replicate path give identical numbers for the same seed (tested), so it is one draw from this distribution, not an artefact.", "",
           f"Why the readout's standard error ({ro['estimate']['se']:.3f}) exceeds the spread of replicate estimates ({rs['estimate_sd']:.3f}; median delta-method SE {rs['median_delta_method_se']:.3f}): the replicates keep the assignment and unit parameters fixed, "
           "so they only vary the observation noise, while the delta-method SE also reflects unit-to-unit heterogeneity as if assignment were random. The one-sided lower bound therefore covers the truth more often than its nominal level (conservative).", "",
           "The planning study is a bootstrap of pre-period noise with a declared effect; this study uses the simulator's own (different) generative model, so agreement between them is a check on the planning model, not a tautology.", "",
           "## 8. Limitations", "",
           "- All of it is simulation: units, outcomes and the truth are synthetic and the simulator's truth model is this project's own assumption; the result says nothing about real incrementality.",
           "- The dev-seed channel (about 100 installs/day) cannot support this test at the declared standards at any declared scale; the policy action for it is insufficient_evidence. The lifecycle above is a demonstration in a SEPARATELY DECLARED, much larger and lower-noise synthetic world, not that channel.",
           "- The demonstration world was declared after the dev-seed result was known, in order to exercise the lifecycle; it is labelled as such and committed before it was run, but it is a judgement call, not an independent test.",
           "- The treated units absorb the whole approved channel step, so their relative increase is about twice the channel's; the measured iROAS belongs to that unit-level step.",
           "- TRUST BOUNDARY: the ledger holds the authoritative experiment configuration and refuses plans that weaken its standards, estimator, guardrails, durations, horizon or unit counts, that record planning numbers "
           "of the wrong shape or for a different design (design fingerprint), or that choose a window the standards would not. It does NOT re-run the planning simulation, so it cannot prove the recorded keep rates were "
           "really produced for this design; the study is deterministic, and re-running it is the independent check.",
           "- The decision function now FAILS CLOSED: a missing, infinite or impossible statistic (any guardrail included) is an invalid readout and means rollback. Unavailable cells (NaN/inf, negative counts, D1-active above installs) "
           "are distinguished from valid zeros.",
           "- Unit-level heavy-tailed revenue and a modest number of units make the t approximation imperfect; see the planning false-keep rates and the replicate-study coverage above.",
           "- One readout is one random draw; keep/rollback on a single experiment is not a verdict on the channel.", "",
           "## 9. Corrections made after independent review (this report was regenerated with them)", "",
           "1. **Readout execution is authoritative.** `execute_readout` no longer accepts a caller's decision. It re-verifies the registered plan, checks that the run link is the active experiment-backed APPLY of this plan "
           "(`start_experiment` verifies it too), uses the experiment store's own clock and observations, recomputes the readout with the production function under explicitly supplied, verified configuration, and executes only that. "
           "A supplied readout is a preview and any mismatch is rejected. Regressions: fabricated early keep and rollback, incomplete observations, a decision edited after a valid readout, a wrong ledger-change link. Refusals record no outcome and change no budget.",
           "2. **The decision fails closed.** `design.decide` treats a missing (NaN), infinite or impossible statistic, guardrails included, as an invalid readout (`rollback:invalid_readout_statistic`), never as a keep. "
           "`aggregate_observations` separates unavailable or impossible cells from valid zeros. The same function serves the planning study and the readout.",
           "3. **Plans are validated against the authoritative experiment configuration.** The ledger holds an immutable snapshot of `config/experiment.yaml` and refuses a plan whose adequacy standards, estimator, alpha, guardrails, "
           "durations, horizon or unit counts differ from it (a plan can no longer lower the required power and keep the original `experiment_config_hash`), whose recorded planning numbers have the wrong shape, impossible Monte Carlo errors, "
           "or were recorded for a different design (design fingerprint), or whose chosen window is not the shortest one the standards accept. The plan content now includes that fingerprint, so its hash differs from the earlier report's.",
           "Remaining trust boundary: the ledger does not re-run the planning study (see Limitations).", ""]
    (ROOT / "reports" / "phase4_report.md").write_text("\n".join(L_), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    main(p.parse_args().seed)
