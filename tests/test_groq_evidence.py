"""The recorded Groq live runs, their hash manifest, and what the README says about them. Offline; no model call. ALL DATA IS SYNTHETIC.
The README must say exactly what the committed records support: one validated live proposal on one review, equal to the deterministic reference, four fallbacks,
no reliability or decision-quality claim."""
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import groq_view as GV  # noqa: E402

README = (ROOT / "README.md").read_text(encoding="utf-8")
RUNS = ["reports/phase5_runs/live_groq_2025-11-10%s.json" % s for s in ("", "_2", "_3", "_4", "_5")]


def test_the_recorded_runs_match_their_manifest():
    v = GV.load()
    assert v is not None and v["problems"] == []
    man = v["manifest"]
    assert set(man["files"]) == set(RUNS) | {"reports/groq_live_status.json", "reports/groq_smoke_status.json"}
    assert all(re.fullmatch(r"[0-9a-f]{40}", c) for c in man["original_development_history"].values() if c != man["original_development_history"]["note"])
    assert not re.search(r"gsk_[A-Za-z0-9]{10,}", "".join((ROOT / r).read_text(encoding="utf-8") for r in man["files"]))  # no key in any record


def test_a_tampered_or_missing_record_is_detected(tmp_path, monkeypatch):
    for rel in list(GV.json.loads(GV.MANIFEST.read_text(encoding="utf-8"))["files"]) + ["reports/ui_review_cache.json"]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, tmp_path / rel)
    shutil.copy(GV.MANIFEST, tmp_path / "reports" / "groq_runs_manifest.json")
    monkeypatch.setattr(GV, "EVIDENCE", tmp_path)
    monkeypatch.setattr(GV, "MANIFEST", tmp_path / "reports" / "groq_runs_manifest.json")
    monkeypatch.setattr(GV, "CACHE", tmp_path / "reports" / "ui_review_cache.json")
    assert GV.load()["problems"] == []
    f = tmp_path / RUNS[3]
    f.write_text(f.read_text(encoding="utf-8").replace('"agent_proposal"', '"agent_proposalX"', 1), encoding="utf-8")
    assert GV.load()["problems"] == [f"changed since recorded: {RUNS[3]}"]
    (tmp_path / "reports" / "groq_smoke_status.json").unlink()
    assert any(p.startswith("missing:") for p in GV.load()["problems"])


def test_the_records_say_what_the_readme_claims():
    v = GV.load()
    assert (v["n_attempts"], v["n_validated"], v["n_fallback"]) == (5, 1, 4)
    a = [x for x in v["attempts"] if x["status"] == "agent_proposal"]
    assert len(a) == 1 and a[0]["attempt"] == 4 and a[0]["mode"] == "live" and a[0]["provider"] == "groq" and a[0]["model"] == "openai/gpt-oss-120b" and a[0]["as_of"] == "2025-11-10"
    assert [x["status"] for x in v["attempts"] if x["attempt"] != 4] == ["rules_only_fallback"] * 4
    assert v["all_match"] and len(v["comparison"]) == 5 and all(c["match"] for c in v["comparison"])
    assert v["tools_used"] == ["get_policy_decision", "get_review_overview"]  # no investigation beyond the policy decisions
    st = v["stages"]
    assert st["investigation"]["required_evidence_complete"] is True and st["finalization"]["validator_ok"] is True and st["finalization"]["repairs_used"] == 0
    cache = json.loads((ROOT / "reports" / "ui_review_cache.json").read_text(encoding="utf-8"))  # the reference is the committed rules-only proposal for that date
    ref = {i["channel"]: (i["action"], i["new_budget"]) for i in cache["dates"]["2025-11-10"]["rules_only"]["proposal"]["items"]}
    assert ref == {c["channel"]: (c["groq_action"], c["groq_budget"]) for c in v["comparison"]}


def test_the_readme_reports_the_groq_evidence_and_its_limits():
    assert "## Live agent evidence (Groq)" in README and "reports/groq_runs_manifest.json" in README
    assert "1 validated live proposal on one synthetic review (2025-11-10) out of 5 full-case attempts; the other 4 fell back" in README
    assert "No reliability and no decision-quality improvement is demonstrated" in README
    assert "Its actions and budgets match the deterministic reference on 5 of 5 channels" in README
    assert "Gemini remains blocked by its separately recorded billing state (HTTP 402)" in README
    assert "Seeds 9101-9103 are consumed by the baseline run and are not untouched agent holdouts" in README
    assert "## Held-out results: the frozen deterministic baseline only" in README
    for s in (9101, 9102, 9103):
        r = json.loads((ROOT / "evals" / "results" / "heldout" / f"seed_{s}" / "baseline_eval.json").read_text(encoding="utf-8"))
        c = next(x for x in r["forecasting"]["table"] if x["level"] == "cohort" and x["group"] == "overall")
        assert f"| {s} | {c['n']} | {c['wape']:.3f} |" in README


def test_the_readme_has_no_unsupported_claims():
    low = README.lower()
    for bad in ("agent outperforms", "outperform", "improves decisions", "business impact:", "production-ready", "in production", "customers"):
        assert bad not in low, bad
    for m in re.finditer(r"\bprov(?:e|es)\b", low):  # whole words only; every use must be a negation
        assert re.search(r"(cannot|not|none of that|no)\s+(?:\w+\s+){0,3}$", low[max(0, m.start() - 40):m.start()]), "a positive 'prove' claim: " + low[max(0, m.start() - 60):m.end() + 30]
    assert "does not show" in low and README.count("validated live proposal") >= 2
