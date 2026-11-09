"""Render reports/heldout_baseline_report.md from the stored held-out results (presentation only: it computes no result). ALL DATA IS SYNTHETIC.

    python scripts/build_heldout_report.py [--results evals/results/heldout] [--out reports/heldout_baseline_report.md]

Every number comes from evals/results/heldout/seed_N/{baseline_eval,provenance}.json; the pooled table sums per-seed COUNTS and says n.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.dev_eval import md  # noqa: E402

BOUNDARY = ("**What this is.** An evaluation of the FROZEN DETERMINISTIC RULES-ONLY BASELINE on three additional synthetic seeds. It validates nothing about any live agent: no model was called, "
            "there is no agent-vs-baseline comparison, and live Gemini verification remains PENDING (blocked by a recorded billing state, HTTP 402). These seeds have now been used for this "
            "evaluation, so they can NOT later be described as untouched holdouts for an agent. All data is synthetic; the ledger is a sandbox; experiments are simulated.")


def load(results: Path) -> dict:
    out = {}
    for d in sorted(results.glob("seed_*")):
        s = int(d.name.split("_")[1])
        prov = json.loads((d / "provenance.json").read_text(encoding="utf-8")) if (d / "provenance.json").exists() else None
        res = json.loads((d / "baseline_eval.json").read_text(encoding="utf-8")) if (d / "baseline_eval.json").exists() else None
        out[s] = {"prov": prov, "res": res}
    return out


def policy_section(p: dict, title: str) -> list:
    o, a = p["overall"], p["abstention"]
    per = pd.DataFrame(p["per_scenario"])
    L = [f"#### {title}", "", f"{o['n_scored']} scored channel-reviews: ideal {o['ideal']}, acceptable {o['acceptable']}, forbidden {o['forbidden']}, unlisted {o['unlisted']} "
         f"(ideal {o['ideal_rate']:.3f}, acceptable-or-better {o['acceptable_or_better_rate']:.3f}, forbidden {o['forbidden_rate']:.3f}).", "", md(per[["scenario", "n_scored", "ideal", "acceptable", "forbidden", "unlisted"]]) if len(per) else "(none)", ""]
    rec, app = a["recall"], a["appropriateness"]
    rec_s, app_s = ("n/a" if rec is None else f"{rec:.3f}"), ("n/a" if app is None else f"{app:.3f}")
    L += [f"Abstentions: recall {rec_s} over {a['n_rows_where_ie_is_the_only_ideal']} rows where insufficient_evidence is the only ideal action; "
          f"{a['n_abstained']} abstained, appropriateness {app_s}; false abstentions {len(a['false_abstentions'])}; missed abstentions {len(a['missed_abstentions'])}.", ""]
    return L


def render(data: dict) -> str:
    L = ["# Held-out evaluation of the frozen deterministic baseline", "", BOUNDARY, ""]
    seeds = sorted(data)
    if not seeds:
        return "\n".join(L + ["No held-out results exist yet."]) + "\n"
    L += ["## Run status", ""]
    rows = []
    for s in seeds:
        pr = data[s]["prov"]
        rows.append({"seed": s, **{n: (pr["steps"].get(n, {}).get("status", "not run") if pr else "no provenance") for n in ("generate", "backtest", "tie_checks", "evaluate")},
                     "freeze": pr["freeze_id"] if pr else "", "commit": (pr["git_commit"][:10] if pr else "")})
    L += [md(pd.DataFrame(rows)), ""]
    for s in seeds:
        pr = data[s]["prov"]
        for n, st in (pr["steps"].items() if pr else []):
            if st["status"] == "failed":
                L += [f"**seed {s}, step {n} FAILED (preserved, not retried):** {st['error']}", ""]
    ok = [s for s in seeds if data[s]["res"]]
    L += ["## Per-seed results", ""]
    for s in ok:
        r, pr = data[s]["res"], data[s]["prov"]
        f = r["forecasting"]
        wb = ", ".join((k + ": " + str(v["cohorts"])) if isinstance(v, dict) and "cohorts" in v else (k + ": " + str(v)) for k, v in f["withheld_by_reason"].items())
        L += [f"### Seed {s}", "", "Artifact hashes (sha256, from provenance.json): " + ", ".join(f"{k} `{v[:12]}`" for k, v in r["artifact_hashes"].items()), "",
              "#### Forecasting (scored against D30 outcomes in storage; nominal interval coverage 0.80)", "",
              md(pd.DataFrame([x for x in f["table"] if x["group"] == "overall" or x["group"].startswith("channel")])), "",
              f"Accounting (cohorts): {f['cohort_accounting']['issued']} issued, {f['cohort_accounting']['scored']} scored, {f['cohort_accounting']['pending_outcome_not_in_storage']} pending, "
              f"{f['cohort_accounting']['withheld_total']} withheld ({wb}); "
              f"share of matured D30 revenue withheld {f['cohort_accounting']['share_of_matured_d30_revenue_withheld']:.3f}.", ""]
        L += policy_section(r["policy_primary"], "Policy and abstentions - PRIMARY (all scored rows, as registered)")
        bad = r["policy_primary"]["forbidden_or_unlisted"]
        L += ["Disagreements with the committed expectations (forbidden or unlisted):", "", md(pd.DataFrame(bad)[["scenario", "channel", "date", "action", "class", "ideal", "acceptable", "forbidden", "reason"]]) if bad else "(none)", ""]
        L += ["Reasons behind every insufficient_evidence / hold (first stated reason, numbers masked):", "", md(pd.DataFrame(r["policy_primary"]["reasons_table"])), ""]
        sv = r["scenario_validity"]
        L += [f"#### Scenario validity: {sv['n_pass']}/{sv['n_intents']} generator calibration intents hold on this seed", ""]
        if sv["scenarios_not_materialized"]:
            L += [f"**Scenarios flagged `scenario_not_materialized` on seed {s}: {', '.join(sv['scenarios_not_materialized'])}.** Failing intents:", "",
                  md(pd.DataFrame([x for x in sv["table"] if not x["pass"]])), ""]
            sec = r["policy_secondary_excluding_unmaterialized_scenarios"]
            L += (policy_section(sec, "Policy and abstentions - SECONDARY (flagged scenarios left out; shown alongside, never instead)") if sec and "error" not in sec else [f"Secondary summary: {sec}", ""])
        else:
            L += ["No scenario was flagged on this seed.", ""]
        L += ["<details><summary>every intent</summary>", "", md(pd.DataFrame(sv["table"])), "", "</details>", ""]
    if ok:
        pol = [data[s]["res"]["policy_primary"]["overall"] for s in ok]
        tot = {k: sum(p[k] for p in pol) for k in ("n_scored", "ideal", "acceptable", "forbidden", "unlisted")}
        L += ["## Pooled counts (sums of the per-seed counts above; not a replacement for them)", "",
              md(pd.DataFrame([{"seeds": ", ".join(map(str, ok)), **tot, "forbidden_rate": tot["forbidden"] / tot["n_scored"], "acceptable_or_better_rate": (tot["ideal"] + tot["acceptable"]) / tot["n_scored"]}])), "",
              "Forecast metrics are not pooled (a ratio of sums cannot be rebuilt from per-seed ratios); per-seed cohort-level overall values:", "",
              md(pd.DataFrame([{"seed": s, **{k: v for k, v in next(x for x in data[s]["res"]["forecasting"]["table"] if x["level"] == "cohort" and x["group"] == "overall").items() if k != "level" and k != "group"}} for s in ok])), ""]
    L += ["## Limitations", "",
          "- Three synthetic seeds from the generator the rules were developed against: this is not evidence about real data or other generators.",
          "- Interval coverage is compared with nominal 0.80; any shortfall above is a finding, not something corrected.",
          "- A `forbidden` row is a violation of a committed scenario expectation, not independently proven economic harm.",
          "- Only the deterministic baseline was run. The constraints suite and experiment safeguards are seed-independent and are reported in the development evaluation.", ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "evals" / "results" / "heldout"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "heldout_baseline_report.md"))
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(render(load(Path(a.results))), encoding="utf-8")
    print("wrote", a.out)
