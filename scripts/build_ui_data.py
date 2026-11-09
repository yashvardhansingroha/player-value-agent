"""Build reports/ui_review_cache.json: for every dev review date, the deterministic policy decisions, the forecast/quality evidence, the rules-only baseline proposal and the
MOCK (scripted) agent run, all from REAL runs of the existing code on the dev seed. ALL DATA IS SYNTHETIC; experiments are SIMULATED; the ledger is a SANDBOX.

    python scripts/build_ui_data.py          # about 1-2 minutes; no network, no Gemini call

The Streamlit screen (app/main.py) only READS this file (plus the committed Phase 4 / Phase 5 / Phase 6 results). It never imports the simulator, never reads hidden truth and
never calls a model. A guard in the app compares the cache's policy hash with the current configuration and refuses to show a stale cache.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_agent import build_environment  # noqa: E402  (harness: it injects the simulator-backed pre-period provider)
from src.agent import agent as A  # noqa: E402
from src.agent import validator as V  # noqa: E402
from src.agent.scripted import scripted_llm  # noqa: E402
from src.ledger.proposal import jsonable, proposal_body  # noqa: E402
from evals.dev_eval import GEN, day_date  # noqa: E402


def channel_view(ctx, ch: str) -> dict:
    d = ctx.decision(ch)
    ev = d["evidence"]
    step = ev.get("marginal_d30_roas_step")
    pops = ev.get("populations")
    return {
        "current_budget": d["current_budget"], "policy_action": d["action"], "policy_budget": d["proposed_budget"], "reasons": list(d["reasons"]),
        "stages": [{"stage": s["stage"], "passed": s["passed"], "blocks": s["blocks"], "detail": s["detail"]} for s in d["stages"]],
        "completeness_passed": next((s["passed"] for s in d["stages"] if s["stage"] == "completeness"), None),
        "forecast": None if pops is None else {
            "window": pops["window"], "included_cohorts": pops["included"]["n_cohorts"], "included_installs": pops["included"]["installs"], "included_spend": pops["included"]["spend"],
            "forecast_d30_net_revenue": pops["included"]["forecast_d30_net_revenue"], "excluded_withheld_cohorts": pops["excluded_withheld"]["n_cohorts"],
            "excluded_withheld_spend": pops["excluded_withheld"]["spend"], "included_spend_share": pops["included_spend_share"], "n_expected_dates": pops["n_expected_dates"],
            "interval": ev.get("uncertainty"), "note": pops["note"]},
        "average_d30_roas": ev.get("average_d30_roas"), "payback_target": ev.get("payback_target"),
        "marginal_step": None if step is None else {"from": step["finite_step"]["s0"], "to": step["finite_step"]["s1"], "marginal_roas": step["finite_step"]["marginal_roas"],
                                                   "lower_bound": step["lower_bound"]["lower"], "label": step["label"]}}


def main():
    t0 = time.time()
    out = {"label": "REAL deterministic runs on the DEV seed (seed 1). ALL DATA IS SYNTHETIC. Agent runs here are MOCK plumbing, never a live model.", "dates": {}}
    for day in GEN["decision_as_of_days"]:
        as_of = day_date(day)
        ctx = build_environment(as_of)
        mock = A.run_agent(ctx, scripted_llm(ctx))
        items = A.rules_only_items(ctx)
        rep = V.validate_rules_items(items, ctx)
        labels = ["RULES-ONLY BASELINE: the deterministic policy after the experiment-design gate (no agent involved).", *A.condition_labels(ctx, items)]
        rules = A.rules_only_proposal(ctx, items, labels)
        out["policy_hash"] = ctx.policy_hash
        out["dates"][as_of] = {
            "as_of": as_of, "ledger_version": ctx.ledger_version, "budgets": ctx.budgets, "ceiling": ctx.ceiling, "policy_hash": ctx.policy_hash,
            "channels": {ch: channel_view(ctx, ch) for ch in ctx.channels},
            "rules_only": {"mode": "rules_only", "labels": labels, "proposal": proposal_body(rules), "validation": rep, "plans_to_register": rep["plans_to_register"]},
            "agent_mock": mock.to_dict()}
        print(f"{as_of}: policy {[ctx.decision(c)['action'] for c in ctx.channels]}; mock {mock.status}; rules-only validated {rep['ok']}", flush=True)
    out["elapsed_seconds"] = round(time.time() - t0, 1)
    (ROOT / "reports" / "ui_review_cache.json").write_text(json.dumps(jsonable(out), indent=1), encoding="utf-8")
    print("wrote reports/ui_review_cache.json")


if __name__ == "__main__":
    main()
