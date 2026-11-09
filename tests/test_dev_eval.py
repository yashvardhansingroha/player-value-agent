"""The Phase 6 development-only evaluation harness: its scoring logic on hand-made inputs (independent of the real data), the pre-registered criteria file, the
held-out guard and the adversarial suite. Offline; no Gemini call. ALL DATA IS SYNTHETIC."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from evals import dev_eval as E

ROOT = Path(__file__).resolve().parents[1]
CRIT = yaml.safe_load((ROOT / "evals" / "criteria.yaml").read_text(encoding="utf-8"))


def fake_reviews(action_for):
    """A reviews dict shaped like run_reviews()'s output: every channel-review gets action_for(channel, date)."""
    chans = ["applovin", "google_uac", "meta", "tiktok", "unity"]
    res = {}
    for d in E.GEN["decision_as_of_days"]:
        dt = E.day_date(d)
        res[dt] = {"decisions": {c: {"action": action_for(c, dt), "reasons": [f"stated reason for {c} on {dt} with number 12.5"]} for c in chans}}
    return {"results": res}


def test_day_to_date_mapping_and_the_class_precedence():
    assert E.day_date(245) == "2025-09-08" and E.day_date(336) == "2025-12-08"
    ex = {"ideal_actions": ["hold"], "acceptable_actions": ["hold", "insufficient_evidence"], "forbidden_actions": ["decrease", "hold"]}
    assert E.classify("hold", ex) == "ideal"  # ideal beats acceptable beats forbidden when a list overlaps
    assert E.classify("insufficient_evidence", ex) == "acceptable" and E.classify("decrease", ex) == "forbidden" and E.classify("increase_test", ex) == "unlisted"


def test_policy_scoring_counts_classes_rates_and_lists_every_disagreement():
    out = E.evaluate_policy(fake_reviews(lambda c, d: "hold"))
    o = out["overall"]
    assert o["n_scored"] == 48 and o["ideal"] + o["acceptable"] + o["forbidden"] + o["unlisted"] == 48 and isinstance(o["n_scored"], int)  # 48 channel-reviews sit inside scenario windows
    assert o["forbidden"] == 0 and o["unlisted"] == 0  # holding is never forbidden in any scenario
    assert o["ideal_rate"] == pytest.approx(o["ideal"] / 48) and o["acceptable_or_better_rate"] == pytest.approx((o["ideal"] + o["acceptable"]) / 48)
    by = {r["scenario"]: r for r in out["per_scenario"]}
    assert by["steady_winner"]["ideal"] == 0 and by["steady_winner"]["acceptable"] == 4  # ideal is increase_test; hold is only acceptable
    assert by["tracking_outage"]["acceptable"] == 3 and by["tracking_outage"]["ideal"] == 0
    assert "regime_shift" not in by  # no decision expectation: not scored
    assert out["n_background"] == 70 - 48 and out["n_channel_reviews"] == 70
    bad = E.evaluate_policy(fake_reviews(lambda c, d: "decrease"))
    assert bad["overall"]["forbidden"] > 0 and len(bad["forbidden_or_unlisted"]) == bad["overall"]["forbidden"] + bad["overall"]["unlisted"]
    assert all(r["class"] in ("forbidden", "unlisted") for r in bad["forbidden_or_unlisted"])


def test_abstention_metrics():
    ie = "insufficient_evidence"
    always = E.evaluate_policy(fake_reviews(lambda c, d: ie))["abstention"]
    assert always["recall"] == 1.0 and always["n_abstained"] == 48 and always["missed_abstentions"] == []
    assert always["false_abstentions"], "steady_winner lists no abstention as ideal or acceptable, so abstaining there is a FALSE abstention"
    assert all(r["scenario"] in ("steady_winner", "front_loaded", "saturating_imperfect_curve", "delayed_reporting", "cpi_spike", "tiny_sample_whale", "tracking_outage") for r in always["false_abstentions"])
    never = E.evaluate_policy(fake_reviews(lambda c, d: "hold"))["abstention"]
    assert never["recall"] == 0.0 and never["n_abstained"] == 0 and never["appropriateness"] is None and never["missed_abstentions"] == []
    cut = E.evaluate_policy(fake_reviews(lambda c, d: "decrease"))["abstention"]
    assert cut["missed_abstentions"] and all("decrease" in r["forbidden"] for r in cut["missed_abstentions"])


def test_reason_masking_groups_reasons_that_differ_only_in_numbers():
    rows = E.evaluate_policy(fake_reviews(lambda c, d: "hold"))["reasons_table"]
    assert rows and all("#" in r["first stated reason (numbers masked)"] or r["first stated reason (numbers masked)"] for r in rows)
    assert sum(r["channel-reviews"] for r in rows) == 70 and not any("12.5" in r["first stated reason (numbers masked)"] for r in rows)


def test_the_adversarial_suite_refuses_every_case_with_the_expected_code():
    cases = E.constraint_cases()
    assert [c["id"] for c in cases] == [f"C{i:02d}" for i in range(1, 20)] == [c["id"] for c in CRIT["components"]["constraints"]["cases"]]  # exactly the pre-registered cases
    assert [c["expected"] for c in cases] == [c["expect"] for c in CRIT["components"]["constraints"]["cases"]]
    assert all(c["refused_as_expected"] for c in cases), [c for c in cases if not c["refused_as_expected"]]


def test_a_case_that_is_not_refused_is_reported_as_such_not_hidden(monkeypatch):
    from src.policy import constraints as cons
    monkeypatch.setattr(cons, "check_constraints", lambda *a, **k: [])  # a broken constraint checker that refuses nothing
    cases = {c["id"]: c for c in E.constraint_cases()}
    assert cases["C01"]["refused_as_expected"] is False and cases["C01"]["observed"] == "(nothing refused)"


def test_criteria_are_pre_registered_complete_and_name_real_tests():
    assert CRIT["version"] == 1 and isinstance(CRIT["amendments"], list)
    assert set(CRIT["components"]) == {"forecasting", "policy_decisions", "abstentions", "constraints", "experiment_safeguards", "agent_plumbing_mock"}  # evaluated SEPARATELY
    assert CRIT["live_agent"]["status"] == "PENDING" and CRIT["held_out"]["seeds"] == E.GEN["heldout_seeds"]
    ids = [i for g in CRIT["components"]["experiment_safeguards"]["groups"].values() for i in g]
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "--no-header", *ids], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0 and "error" not in out.stdout.lower() and "tests collected" in out.stdout  # every named safeguard test exists
    assert set(CRIT["components"]["agent_plumbing_mock"]["investigation_coverage"]["mapping"].values()) <= {"get_quality_checks", "get_forecast_and_populations", "get_economics", "plan_experiment"}


def test_scenario_required_investigations_are_all_mapped():
    mapping = CRIT["components"]["agent_plumbing_mock"]["investigation_coverage"]["mapping"]
    for p in (ROOT / "evals" / "scenarios").glob("*.json"):
        for r in json.loads(p.read_text(encoding="utf-8"))["expected"].get("required_investigations") or []:
            assert r in mapping, (p.name, r)


def test_held_out_guard_holds_and_the_development_harness_cannot_use_held_out_seeds():
    h = E.held_out_status()
    assert h["generator_refuses_without_allow_flag"] is True  # in every state, the generator refuses a held-out seed without the explicit flag
    if not (ROOT / "evals" / "results" / "heldout" / "RUN_LOCK.json").exists():
        assert h["touched"] is False and h["data_present_for_heldout_seeds"] == []  # before the baseline run the seeds are untouched
    for f in ("evals/dev_eval.py", "scripts/run_dev_eval.py"):
        text = (ROOT / f).read_text(encoding="utf-8")
        assert "allow_heldout=True" not in text and "--allow-heldout" not in text  # the development harness can neither generate nor score a held-out seed


def test_live_status_is_pending_unless_a_validated_live_agent_run_exists_and_failed_attempts_are_kept():
    s = E.live_status()
    valid = sum(1 for a in s["attempts"] for r in a.get("live_agent_runs", []) if r.get("status") == "agent_proposal")
    assert (s["status"] == "PENDING") == (valid == 0) and s["valid_live_agent_runs"] == valid
    assert len(s["attempts"]) >= 4 and s["attempts"][0]["connectivity_check"].startswith("PASSED")  # attempts 1-4 are preserved, attempt 1 with its recorded connectivity result


HAVE_DATA = all((ROOT / "data" / p).exists() for p in ("decision/seed_1/decision.db", "forecasts/seed_1/forecasts.db"))


@pytest.mark.skipif(not HAVE_DATA, reason="generated dev data not present (data/ is gitignored)")
def test_forecasting_follows_the_registered_definitions_mean_width_all_breakdowns_and_stated_applicability():
    import sqlite3
    dec = sqlite3.connect(ROOT / "data" / "decision" / "seed_1" / "decision.db")
    st = sqlite3.connect(ROOT / "data" / "forecasts" / "seed_1" / "forecasts.db")
    f = E.evaluate_forecasting(dec, st)
    overall = next(r for r in f["table"] if r["level"] == "cohort" and r["group"] == "overall")
    # the two numbers computed independently by the reviewer for the cohort intervals
    assert overall["interval_width_over_forecast_mean"] == pytest.approx(0.8569837907383542, rel=1e-12)
    assert overall["interval_width_over_forecast_median"] == pytest.approx(0.8876262817740225, rel=1e-12)
    chk = f["independent_width_check"]
    assert chk["mean_matches_scorer"] and chk["median_matches_scorer"] and chk["mean_differs_from_median"] and chk["scorer_mean"] != chk["scorer_median"]
    assert "interval_width_over_forecast" not in overall  # the registered name is not used for the median any more
    groups = {(r["level"], r["group"].split(" ")[0]) for r in f["table"]}
    assert {(lv, b) for lv in ("cohort", "weekly_summary") for b in ("overall", "channel", "method", "period")} <= groups  # ALL four registered breakdowns, at both levels
    na = f["breakdown_applicability"]
    assert all(v == "applies" for v in na["cohort"].values()) and all(v.startswith("applies") for v in na["weekly_summary"].values())
    assert f["summaries_with_mixed_methods"][1] == 239 and f["summaries_with_mixed_methods"][0] == 0  # measured: every summary has a single method, so its method breakdown is well defined
    assert {"cohort / method pooled_multiplier"} <= set(f["groups_without_any_interval_so_coverage_and_width_are_not_applicable"])  # no interval exists there: coverage/width stated as not applicable
    assert all(f"{r['level']} / {r['group']}" in f["groups_without_any_interval_so_coverage_and_width_are_not_applicable"] or r["coverage"] is not None for r in f["table"])


def test_the_tool_budget_amendment_is_quoted_from_the_config_and_criteria_amendments_are_listed():
    tb = E.tool_budget_amendment()
    assert tb["max_tool_calls_in_force"] == 40 and "was 20" in tb["recorded_note"] and "NOT raised to obtain a particular outcome" in tb["recorded_note"]
    assert [a["kind"] for a in CRIT["amendments"]] and all("no metric, class, case, threshold or definition changed" in a["kind"] for a in CRIT["amendments"])
    assert "UNKNOWN" in CRIT["amendments"][0]["what"] and "meta" in CRIT["amendments"][0]["what"]


def test_the_report_text_describes_forbidden_rows_as_scenario_violations_and_does_not_infer_other_channels_experiments():
    src = (ROOT / "scripts" / "run_dev_eval.py").read_text(encoding="utf-8")
    assert "violates a committed scenario expectation" in src and "NOT independent proof that the action would have caused economic harm" in src
    assert "UNKNOWN: no study of it exists" in src and "including meta" in src
    assert "Nothing shows yet that the live investigation problem is resolved" in src
    report = ROOT / "reports" / "phase6_dev_report.md"
    if report.exists():
        text = report.read_text(encoding="utf-8")
        assert all("UNKNOWN" in l for l in text.splitlines() if "meta" in l.lower() and "adequa" in l.lower())  # any sentence linking meta and (in)adequacy says UNKNOWN
        assert "interval_width_over_forecast |" not in text  # the old mislabelled column is gone


def test_the_dev_manifest_is_labelled_as_not_a_freeze():
    m = E.dev_manifest()
    assert m["NOT A FREEZE"] and "agent system prompt" in m["hashes"] and "tool declarations" in m["hashes"] and "evals/criteria.yaml" in m["hashes"]
    assert all(len(v) == 64 for v in m["hashes"].values())
