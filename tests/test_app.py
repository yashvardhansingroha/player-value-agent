"""The Monday-review screen (app/main.py) run headless with Streamlit's AppTest, plus its logic module. It reads the COMMITTED cache (reports/ui_review_cache.json) and result files, so
it needs no generated dev data and makes no model call. ALL DATA IS SYNTHETIC; the ledger is a SANDBOX."""
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app import ui_logic as U

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "main.py")


def run(day=None, source=None):
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert not at.exception, [e.value for e in at.exception]
    if day:
        at.selectbox(key="day").set_value(day).run()
    if source:
        at.radio(key="source").set_value(source).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def click(at, key):
    at.button(key=key).click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def texts(at):
    return " | ".join(x.value for x in list(at.warning) + list(at.info) + list(at.success) + list(at.error) + list(at.caption) + list(at.markdown))


def test_the_screen_runs_shows_the_synthetic_banner_six_tabs_and_live_verification_as_pending():
    at = run()
    assert [t.label for t in at.tabs] == ["1. Monday review", "2. Forecasts and quality", "3. Proposal and evidence", "4. Approval and ledger", "5. Experiments (SIMULATED)", "6. Evaluation"]
    banner = at.warning[0].value
    assert "SYNTHETIC DATA" in banner and "SIMULATED EXPERIMENTS" in banner and "SANDBOX LEDGER" in banner
    info = " ".join(i.value for i in at.info)
    assert "Live agent verification: Gemini PENDING" in info and "HTTP 402" in info and not at.error
    assert "Groq (openai/gpt-oss-120b" in info and "1 validated live proposal on one synthetic review out of 5 full-case attempts" in info and "No reliability or decision-quality improvement is demonstrated" in info
    assert "not that live run" in info  # the screen itself shows rules-only and MOCK proposals only


def test_the_default_review_shows_the_inadequate_experiment_instead_of_an_increase_and_labels_the_baseline():
    at = run("2025-10-13")
    allw = " | ".join(w.value for w in at.warning) + " | ".join(i.value for i in at.info)
    assert "INADEQUATE EXPERIMENT: applovin" in allw and "RULES-ONLY BASELINE" in allw
    assert "Validator: OK" in " ".join(s.value for s in at.success)
    prop = U.get_source(U.load_cache(), "2025-10-13", "rules_only")["proposal"]
    app = next(i for i in prop["items"] if i["channel"] == "applovin")
    assert app["action"] == "insufficient_evidence" and app["new_budget"] == app["current_budget"] and app["experiment_plan_hash"] is None


def test_the_mock_agent_source_is_labelled_mock_and_shows_its_evidence_trail():
    at = run("2025-11-10", "agent_mock")
    warn = " | ".join(w.value for w in at.warning)
    assert "MODE: MOCK" in warn and "NOT evidence about any model" in warn
    src = U.get_source(U.load_cache(), "2025-11-10", "agent_mock")
    assert src["status"] == "agent_proposal" and src["tool_calls"] and all(c["tool_call_id"].startswith("tc_") for c in src["tool_calls"])
    g = next(i for i in src["proposal"]["items"] if i["channel"] == "google_uac")
    assert g["action"] == "insufficient_evidence" and any(c["field"].endswith(".passed") and c["value"] is False for c in g["evidence"]["claims"])  # it cites the failed tracking-gap gate


def test_a_person_can_submit_approve_apply_and_roll_back_a_decrease_in_the_sandbox_ledger():
    at = run("2025-10-06")  # the policy decreases tiktok on this review
    led = U.open_ledger(at.session_state["sandbox"])
    before = led.state()["budgets"]
    assert before["tiktok"] == 85.0 and led.head() == 1
    click(at, "submit")
    click(at, "approve")
    click(at, "apply")
    led = U.open_ledger(at.session_state["sandbox"])
    after = led.state()
    assert led.head() == 2 and after["kind"] == "APPLY" and after["budgets"]["tiktok"] == pytest.approx(76.5) and after["budgets"]["meta"] == before["meta"]
    assert all(ok for ok, _ in at.session_state["messages"]), at.session_state["messages"]
    at.selectbox(key="rollback_pick")  # the applied change is offered for rollback
    click(at, "rollback")
    led = U.open_ledger(at.session_state["sandbox"])
    assert led.head() == 3 and led.state()["budgets"] == before  # restored exactly


def test_after_one_change_other_proposals_are_refused_as_stale_and_a_hold_only_proposal_has_nothing_to_apply():
    at = run("2025-10-06")
    click(at, "submit")
    click(at, "approve")
    click(at, "apply")
    at.selectbox(key="day").set_value("2025-10-20").run()  # another review, built on ledger version 1
    click(at, "submit")
    ok, msg = at.session_state["messages"][-1]
    assert ok is False and "built on ledger version 1" in msg and "ledger is at version 2" in msg
    at.button(key="reset").click().run()  # a fresh sandbox
    at.selectbox(key="day").set_value("2025-09-08").run()  # everything holds
    click(at, "submit")
    click(at, "approve")
    ok, msg = at.session_state["messages"][-1]
    assert ok is False and "no_budget_change" in msg and U.open_ledger(at.session_state["sandbox"]).head() == 1


def test_a_rejected_proposal_cannot_then_be_applied_and_a_duplicate_submission_is_refused():
    at = run("2025-10-06")
    click(at, "submit")
    click(at, "submit")
    assert at.session_state["messages"][-1][0] is False and "already in the ledger" in at.session_state["messages"][-1][1]
    click(at, "reject")
    click(at, "apply")
    ok, msg = at.session_state["messages"][-1]
    assert ok is False and "not_approved" in msg and U.open_ledger(at.session_state["sandbox"]).head() == 1


def test_a_stale_cache_is_refused_not_shown(monkeypatch):
    cache = U.load_cache()
    stale = dict(cache, policy_hash="0" * 64)
    monkeypatch.setattr(U, "load_cache", lambda: stale)
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert any("DIFFERENT policy/forecast configuration" in e.value for e in at.error) and not at.tabs
    monkeypatch.setattr(U, "load_cache", lambda: None)
    at2 = AppTest.from_file(APP, default_timeout=90).run()
    assert any("missing" in e.value for e in at2.error)


def test_the_experiment_and_evaluation_tabs_say_what_is_known_and_unknown():
    at = run()
    md = " | ".join(x.value for x in at.markdown)
    assert "UNKNOWN" in md and "applovin" in md and "meta" in md  # adequacy of other channels is unknown
    assert "forbidden" in md and "not independent proof of economic harm" in md and "UNDER nominal" in md


def test_the_review_cache_is_current_complete_and_consistent_with_the_committed_phase_results():
    cache = U.load_cache()
    ok, msg = U.cache_status(cache)
    assert ok, msg
    assert len(cache["dates"]) == 14 and U.review_dates(cache)[0] == "2025-09-08" and U.review_dates(cache)[-1] == "2025-12-08"
    for day in cache["dates"]:
        d = cache["dates"][day]
        assert d["rules_only"]["validation"]["ok"] and d["agent_mock"]["mode"] == "mock" and d["agent_mock"]["status"] == "agent_proposal" and d["agent_mock"]["validation"]["ok"]
        assert len(U.review_table(cache, day)) == 5 and len(U.forecast_table(cache, day)) == 5
    p3 = U.read_json(ROOT / "reports" / "phase3_policy_decisions.json")["results"]
    for day, r in p3.items():  # the cache's policy actions are the Phase 3 reviewed decisions
        assert {c: cache["dates"][day]["channels"][c]["policy_action"] for c in r["decisions"]} == {c: d["action"] for c, d in r["decisions"].items()}


def test_the_ui_logic_sandbox_matches_the_ledger_rules(tmp_path):
    cache = U.load_cache()
    sb = U.new_sandbox(cache)
    led = U.open_ledger(sb)
    assert led.head() == 1 and led.state()["budgets"] == cache["dates"]["2025-09-08"]["budgets"]
    prop = U.get_source(cache, "2025-10-06", "rules_only")["proposal"]
    assert U.proposal_status(led, prop["proposal_id"]) is None and U.submit(led, prop, "2025-10-06")["ok"] and U.proposal_status(led, prop["proposal_id"]) == "proposed"
    assert U.decide(led, prop["proposal_id"], "approve", "reviewer", "2025-10-06")["ok"] and U.decide(led, prop["proposal_id"], "apply", "reviewer", "2025-10-06")["ok"]
    ch = U.active_changes(led)
    assert len(ch) == 1 and U.rollback(led, ch[0]["change_id"], "reviewer", "2025-10-07")["ok"] and led.head() == 3
    assert U.active_changes(led) == [] and "rolled_back" in set(r[0] for r in led.conn.execute("SELECT status FROM applied_changes"))


def test_the_app_never_reads_a_key_calls_a_model_or_touches_hidden_data():
    banned_modules = ("src.agent.llm", "src.agent.agent", "src.agent.scripted", "src.generator", "src.experiments.simulator", "src.experiments.scoring", "google", "genai", "subprocess", "os")
    banned_words = ("GEMINI_API_KEY", "os.environ", "getenv", "hidden_truth", "evals/scenarios", "data/hidden", "data\\\\hidden", "GeminiClient", "sqlite3.connect")
    for p in (ROOT / "app").glob("*.py"):
        text = p.read_text(encoding="utf-8")
        code = re.sub(r'""".*?"""', "", "\n".join(l for l in text.splitlines() if not l.strip().startswith("#")), flags=re.DOTALL)
        imported = re.findall(r"^\s*(?:from|import)\s+([\w.]+)", code, flags=re.MULTILINE)
        assert not [m for m in imported if m in banned_modules or m.split(".")[0] in ("google", "genai", "subprocess", "os")], (p.name, imported)
        for bad in banned_words:
            assert bad not in code, (p.name, bad)
