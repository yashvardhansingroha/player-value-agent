"""The held-out DETERMINISTIC-BASELINE harness: freeze verification, the seed/artifact ties, the run-once guards and the scenario-validity flagging. Offline; no model call.
ALL DATA IS SYNTHETIC. These tests use hand-made inputs and the small generated dataset; they never generate or read a held-out seed."""
import copy
import json
import re
import sqlite3
from pathlib import Path

import pytest
import yaml

from evals import baseline_eval as B
from evals import dev_eval as E
from evals import freeze as F
from scripts import run_heldout_baseline as R
from src.forecast import store
from src.forecast.pipeline import load_config, run_backtest

ROOT = Path(__file__).resolve().parents[1]
GEN = yaml.safe_load((ROOT / "config" / "generator.yaml").read_text(encoding="utf-8"))
PROTO = yaml.safe_load((ROOT / "evals" / "heldout_protocol.yaml").read_text(encoding="utf-8"))
CRIT = yaml.safe_load((ROOT / "evals" / "criteria.yaml").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------- protocol and amendment


def test_the_scope_amendment_is_appended_and_does_not_authorise_agent_claims():
    ams = CRIT["amendments"]
    assert [a["date"] for a in ams][:1] == ["2026-11-07"], "the earlier amendment is preserved, in order"
    a = next(x for x in ams if x["date"] == "2026-11-08")
    assert "DETERMINISTIC-BASELINE-ONLY" in a["what"] and "does NOT authorize any claim about live-agent performance" in a["what"]
    assert "The freeze has NOT happened" in CRIT["held_out"]["freeze_required_before_use"], "the original wording is not edited"


def test_the_protocol_registers_the_config_seeds_and_nothing_else():
    assert PROTO["seeds"]["registered"] == GEN["heldout_seeds"] == [9101, 9102, 9103]
    assert set(PROTO["seeds"]["registered"]).isdisjoint(GEN["dev_seeds"]) and PROTO["seeds"]["development"] == GEN["dev_seeds"]
    assert PROTO["freeze_id"] == F.FREEZE_ID and PROTO["seeds"]["run_once"] is True
    assert all(isinstance(x, str) for x in PROTO["what_is_frozen"] + PROTO["artifact_ties"]["checks"] + PROTO["reporting_rules"] + PROTO["failure_and_rerun_rules"])


def test_every_artifact_path_contains_the_seed_and_runs_never_share_a_folder():
    a, b = B.seed_paths(9101), B.seed_paths(9102)
    for k in a:
        assert "seed_9101" in str(a[k]) and "seed_9102" in str(b[k]) and a[k] != b[k]
    assert "seed_1" not in "".join(str(v) for v in a.values())


# ---------------------------------------------------------------------------------------------- freeze verification


def test_frozen_files_cover_the_sources_configs_definitions_and_the_runner():
    fl = set(F.frozen_files())
    for must in ("src/forecast/pipeline.py", "src/policy/policy.py", "config/policy.yaml", "config/forecast.yaml", "config/generator.yaml", "evals/criteria.yaml",
                 "evals/heldout_protocol.yaml", "evals/baseline_eval.py", "evals/dev_eval.py", "evals/freeze.py", "scripts/run_heldout_baseline.py", "scripts/generate_data.py",
                 "evals/scenarios/steady_winner.json", "sql/schema.sql", "requirements.txt"):
        assert must in fl, must
    assert not any("__pycache__" in p or p.startswith(("tests/", "app/", "reports/", "evals/results/")) for p in fl)


def test_hash_ignores_line_endings_for_text_but_not_for_data(tmp_path):
    a, b = tmp_path / "a.py", tmp_path / "b.py"
    a.write_bytes(b"x = 1\r\ny = 2\r\n"), b.write_bytes(b"x = 1\ny = 2\n")
    assert F.file_hash(a) == F.file_hash(b)
    c = tmp_path / "c.db"
    c.write_bytes(b"a\r\nb")
    assert F.plain_sha256(c) != F.plain_sha256(b)


def test_verify_freeze_detects_a_changed_added_or_missing_frozen_file(monkeypatch):
    monkeypatch.setattr(F, "git", lambda *a, **k: "0" * 40)  # no git history is needed to build a manifest for this test
    m = F.build_manifest({"x": 1})
    assert F.verify_freeze(m, check_tag=False, check_dev_artifacts=False) == []
    bad = copy.deepcopy(m)
    bad["frozen_files"]["config/policy.yaml"] = "0" * 64
    assert F.verify_freeze(bad, check_tag=False, check_dev_artifacts=False) == ["frozen file changed since the freeze: config/policy.yaml"]
    short = copy.deepcopy(m)
    del short["frozen_files"]["evals/dev_eval.py"]
    assert any("not in the manifest: evals/dev_eval.py" in p for p in F.verify_freeze(short, check_tag=False, check_dev_artifacts=False))
    gone = copy.deepcopy(m)
    gone["frozen_files"]["src/does_not_exist.py"] = "0" * 64
    assert any("frozen file missing" in p for p in F.verify_freeze(gone, check_tag=False, check_dev_artifacts=False))
    prompt = copy.deepcopy(m)
    prompt["agent_surface"]["system_prompt_sha256"] = "0" * 64
    assert any("agent surface changed" in p for p in F.verify_freeze(prompt, check_tag=False, check_dev_artifacts=False))


def test_verify_freeze_detects_modified_development_artifacts():
    m = F.build_manifest({"x": 1})
    bad = copy.deepcopy(m)
    bad["development_artifacts_seed_1"]["decision_db"] = "0" * 64
    if (ROOT / "data" / "decision" / "seed_1" / "decision.db").exists():
        assert any("development artifact changed" in p for p in F.verify_freeze(bad, check_tag=False))


def test_verify_freeze_without_a_tag_reports_it(monkeypatch):
    m = F.build_manifest({"x": 1})
    m["freeze_id"] = "no-such-tag-for-this-test"
    assert any("missing or not an annotated tag" in p for p in F.verify_freeze(m, check_tag=True, check_dev_artifacts=False))


def test_the_manifest_records_agent_surface_seeds_and_the_rehearsal():
    m = F.build_manifest({"all_reproduced": True})
    assert set(m["agent_surface"]) == {"system_prompt_sha256", "tool_declarations_sha256"}
    assert m["seeds"]["registered_held_out"] == [9101, 9102, 9103] and m["seeds"]["development"] == [1]
    assert m["dev_rehearsal_reproduces"] == {"all_reproduced": True} and "authorises no claim about any live agent" in m["label"]


# ---------------------------------------------------------------------------------------------- run-once guards


def test_the_runner_refuses_when_the_freeze_does_not_verify(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(R, "RES", tmp_path), monkeypatch.setattr(R, "LOCK", tmp_path / "RUN_LOCK.json"), monkeypatch.setattr(R, "LOG", tmp_path / "RUN_LOG.json")
    monkeypatch.setattr(F, "verify_freeze", lambda *a, **k: ["frozen file changed since the freeze: config/policy.yaml"])
    monkeypatch.setattr(R, "build", lambda *a, **k: pytest.fail("data must not be generated when the freeze does not verify"))
    assert R.execute(resume=False) == 2
    assert not (tmp_path / "RUN_LOCK.json").exists() and "REFUSING" in capsys.readouterr().out


def test_a_second_run_is_refused_and_resume_needs_the_same_freeze(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(R, "RES", tmp_path), monkeypatch.setattr(R, "LOCK", tmp_path / "RUN_LOCK.json"), monkeypatch.setattr(R, "LOG", tmp_path / "RUN_LOG.json")
    manifest = {"freeze_id": "baseline-freeze-v1", "seeds": {"registered_held_out": [9101, 9102, 9103]}}
    monkeypatch.setattr(F, "verify_freeze", lambda *a, **k: [])
    monkeypatch.setattr(F, "load_manifest", lambda: manifest)
    monkeypatch.setattr(R, "build", lambda *a, **k: pytest.fail("data must not be generated by a refused run"))
    (tmp_path / "RUN_LOCK.json").write_text(json.dumps({"freeze_id": "baseline-freeze-v1"}))
    assert R.execute(resume=False) == 2 and "already been started" in capsys.readouterr().out
    (tmp_path / "RUN_LOCK.json").write_text(json.dumps({"freeze_id": "some-other-freeze"}))
    assert R.execute(resume=True) == 2
    (tmp_path / "RUN_LOCK.json").unlink()
    assert R.execute(resume=True) == 2, "--resume with no lock is refused"


def test_seeds_run_in_registered_order_each_with_its_own_folders_and_one_failure_does_not_stop_the_rest(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "RES", tmp_path), monkeypatch.setattr(R, "LOCK", tmp_path / "RUN_LOCK.json"), monkeypatch.setattr(R, "LOG", tmp_path / "RUN_LOG.json")
    monkeypatch.setattr(F, "verify_freeze", lambda *a, **k: [])
    monkeypatch.setattr(F, "load_manifest", lambda: {"freeze_id": "baseline-freeze-v1", "seeds": {"registered_held_out": [9101, 9102, 9103]}})
    seen = []

    def fake_build(seed=None, out_root=None, **k):
        seen.append((seed, k.get("allow_heldout")))
        raise RuntimeError(f"stop after recording seed {seed}")

    monkeypatch.setattr(R, "build", fake_build)
    monkeypatch.setattr(R, "ROOT", tmp_path)  # data would be written under tmp_path, never under the real data/
    assert R.execute(resume=False) == 0
    assert seen == [(9101, True), (9102, True), (9103, True)]
    for s in (9101, 9102, 9103):
        prov = json.loads((tmp_path / f"seed_{s}" / "provenance.json").read_text(encoding="utf-8"))
        assert prov["seed"] == s and prov["steps"]["generate"]["status"] == "failed" and "stop after recording" in prov["steps"]["generate"]["error"]
        assert set(prov["steps"]) == {"generate"}, "later steps do not run after a failed step"
        assert prov["completed"] is False
    log = json.loads((tmp_path / "RUN_LOG.json").read_text(encoding="utf-8"))
    assert [e.get("event") for e in log if "event" in e] == ["start", "finished"]


def test_only_the_runner_may_pass_allow_heldout():
    offenders = [str(p.relative_to(ROOT)) for folder in ("src", "app", "evals", "scripts") for p in (ROOT / folder).rglob("*.py")
                 if "allow_heldout" in p.read_text(encoding="utf-8") and p.name not in ("run_heldout_baseline.py", "generate_data.py")]
    assert offenders == []
    runner = (ROOT / "scripts" / "run_heldout_baseline.py").read_text(encoding="utf-8")
    assert "from src.forecast.pipeline import load_config, run_backtest" in runner and "subprocess" not in runner, "the runner calls the pipeline directly, never scripts/run_backtest.py"


# ---------------------------------------------------------------------------------------------- artifact ties


@pytest.fixture(scope="module")
def tied(small_dataset, tmp_path_factory):
    cfg = load_config()
    cfg["quality"].update(min_installs_per_cohort=8, window_min_installs=40)
    cfg["backtest"]["last_as_of_day"] = 168
    f = tmp_path_factory.mktemp("tie") / "f.db"
    run_backtest(small_dataset["db"], f, cfg)
    return {"store": f, "dec": small_dataset["db"], "out": small_dataset["out"]}


def test_content_tie_passes_when_the_store_was_built_from_this_database(tied):
    dec, st = sqlite3.connect(tied["dec"]), sqlite3.connect(tied["store"])
    r = B.content_tie(dec, st, min_rows=20, min_reviews=4)
    assert r["passed"] and r["n_checked"] >= 20 and r["n_reviews"] >= 4 and r["mismatches"] == []


def test_content_tie_fails_when_the_store_is_scored_against_a_different_database(tied, dataset):
    """A forecast store built from one dataset cannot be paired with another seed/scale: the stored features do not match what that database says."""
    other, st = sqlite3.connect(dataset["db"]), sqlite3.connect(tied["store"])
    r = B.content_tie(other, st, min_rows=20, min_reviews=4)
    assert not r["passed"] and r["n_mismatches"] > 0 and r["mismatches"][0]["differing_features"]


def test_content_tie_needs_enough_reviews(tied):
    r = B.content_tie(sqlite3.connect(tied["dec"]), sqlite3.connect(tied["store"]), min_rows=20, min_reviews=500)
    assert not r["passed"] and "reviews with issued forecasts" in r["reason"]


def test_tie_checks_name_each_way_an_artifact_can_be_mispaired(tied, small_dataset):
    paths = B.seed_paths(9101, data_root=tied["out"] / "none")
    assert any("missing artifact" in p for p in B.tie_checks(9101, paths, {}))
    # a dataset that states seed 1 presented as held-out seed 9101, with a store from a different configuration and no recorded hashes
    paths = {"decision_db": tied["dec"], "hidden_truth": Path(tied["out"]) / "hidden" / "seed_1" / "hidden_truth.json", "generation_report": Path(tied["out"]) / "decision" / "seed_1" / "generation_report.json",
             "forecast_store": tied["store"]}
    problems = B.tie_checks(9101, paths, {})
    text = " | ".join(problems)
    assert "hidden_truth.json states seed 1, expected 9101" in text and "generation_report.json states seed 1, expected 9101" in text
    assert "hash at evaluation time differs" in text and "evidence hash differs" in text and "schedule hash differs" in text
    assert any("not a registered held-out seed" in p for p in B.tie_checks(5, paths, {}))
    assert any("is a development seed" in p for p in B.tie_checks(1, paths, {}))
    assert any("rehearsal may only use a development seed" in p for p in B.tie_checks(9101, paths, {}, rehearsal=True))


def test_tie_checks_reject_a_held_out_artifact_identical_to_the_dev_artifact(tied, monkeypatch):
    from evals import freeze
    h = {k: freeze.plain_sha256(p) for k, p in (("decision_db", tied["dec"]), ("forecast_store", tied["store"]),
                                                  ("hidden_truth", Path(tied["out"]) / "hidden" / "seed_1" / "hidden_truth.json"))}
    monkeypatch.setattr(freeze, "data_artifact_hashes", lambda seed, root=None: dict(h))
    paths = {"decision_db": tied["dec"], "hidden_truth": Path(tied["out"]) / "hidden" / "seed_1" / "hidden_truth.json", "generation_report": Path(tied["out"]) / "decision" / "seed_1" / "generation_report.json",
             "forecast_store": tied["store"]}
    assert any("byte-identical to the development seed's artifact" in p for p in B.tie_checks(9101, paths, {}))


# ---------------------------------------------------------------------------------------------- scenario validity and the secondary summary


def test_scenario_validity_flags_a_scenario_with_any_failed_intent_and_keeps_the_primary_summary_whole(monkeypatch):
    rows = [("steady_winner", "m1", 1.0, ">=", 0.5, True), ("steady_winner", "m2", 0.1, ">=", 0.5, False), ("cpi_spike", "m3", 2.0, ">=", 1.0, True)]
    monkeypatch.setattr(B, "compute_intent_metrics", lambda dec, hidden: {})
    monkeypatch.setattr(B, "evaluate_intents", lambda metrics, scenarios: rows)
    v = B.scenario_validity(None, {})
    assert v["scenarios_not_materialized"] == ["steady_winner"] and v["n_intents"] == 3 and v["n_pass"] == 2
    assert [r["pass"] for r in v["table"]] == [True, False, True] and v["table"][1]["value"] == 0.1


def fake_reviews(action):
    chans = ["applovin", "google_uac", "meta", "tiktok", "unity"]
    return {"results": {E.day_date(d): {"decisions": {c: {"action": action, "reasons": ["a reason 1.5"]} for c in chans}} for d in GEN["decision_as_of_days"]}}


def test_excluding_unmaterialized_scenarios_changes_only_the_secondary_summary():
    full = E.evaluate_policy(fake_reviews("hold"))
    sec = E.evaluate_policy(fake_reviews("hold"), exclude_scenarios=["steady_winner"])
    assert full["overall"]["n_scored"] == 48 and sec["overall"]["n_scored"] == 44
    assert "steady_winner" not in {r["scenario"] for r in sec["per_scenario"]} and "steady_winner" in {r["scenario"] for r in full["per_scenario"]}
    assert sec["n_background"] == full["n_background"], "excluded rows are neither scored nor counted as background"
    assert E.evaluate_policy(fake_reviews("hold"))["overall"] == full["overall"], "the default call is unchanged"


def test_evaluate_seed_gives_both_summaries_when_a_scenario_is_flagged(monkeypatch):
    monkeypatch.setattr(B.E, "evaluate_forecasting", lambda dec, st: {"table": []})
    monkeypatch.setattr(B.E, "run_reviews", lambda dec, st: {**fake_reviews("hold"), "budgets": {"applovin": 1.0}, "ceiling": 2.0})
    monkeypatch.setattr(B, "scenario_validity", lambda dec, hidden: {"scenarios_not_materialized": ["steady_winner"], "table": [], "n_intents": 0, "n_pass": 0})
    res = B.evaluate_seed(9101, None, None, {})
    assert res["seed"] == 9101 and res["policy_primary"]["overall"]["n_scored"] == 48
    assert res["policy_secondary_excluding_unmaterialized_scenarios"]["overall"]["n_scored"] == 44
    assert len(res["review_log"]) == 70 and "says nothing about any live agent" in res["label"]
    monkeypatch.setattr(B, "scenario_validity", lambda dec, hidden: {"scenarios_not_materialized": [], "table": [], "n_intents": 0, "n_pass": 0})
    assert B.evaluate_seed(9101, None, None, {})["policy_secondary_excluding_unmaterialized_scenarios"] is None


# ---------------------------------------------------------------------------------------------- state-dependent checks (valid before AND after the freeze / the run)


def test_every_difference_from_the_recorded_freeze_is_declared_and_current():
    # This repository has a fresh history, so the annotated tag is not available; the file hashes are checked instead, and every difference must be declared (evals/publication_delta.json).
    from scripts import verify_baseline_provenance as V
    assert V.check() == []


def test_an_undeclared_change_to_a_frozen_file_is_reported(monkeypatch):
    from scripts import verify_baseline_provenance as V
    real = F.verify_freeze
    monkeypatch.setattr(F, "verify_freeze", lambda **k: real(**k) + ["frozen file changed since the freeze: config/policy.yaml"])
    assert V.check() == ["frozen file changed since the freeze: config/policy.yaml"]


def test_the_manifest_still_records_the_freeze_that_produced_the_held_out_results():
    m = F.load_manifest()
    assert m["freeze_id"] == "baseline-freeze-v1" and len(m["frozen_files"]) == 78 and m["seeds"]["registered_held_out"] == [9101, 9102, 9103]
    lock = json.loads((ROOT / "evals" / "results" / "heldout" / "RUN_LOCK.json").read_text(encoding="utf-8"))
    assert lock["freeze_id"] == m["freeze_id"]


def test_run_state_is_consistent_lock_requires_manifest_and_results_require_lock():
    lock = ROOT / "evals" / "results" / "heldout" / "RUN_LOCK.json"
    if lock.exists():
        assert F.MANIFEST.exists() and json.loads(lock.read_text(encoding="utf-8"))["freeze_id"] == F.load_manifest()["freeze_id"]
    else:
        assert not list((ROOT / "evals" / "results" / "heldout").glob("seed_*")) if (ROOT / "evals" / "results" / "heldout").exists() else True


def test_the_report_renders_from_stored_results_and_states_the_boundary(tmp_path):
    from scripts import build_heldout_report as HR
    reh = ROOT / "evals" / "results" / "rehearsal"
    if not (reh / "seed_1" / "baseline_eval.json").exists():
        pytest.skip("no rehearsal output")
    text = HR.render(HR.load(reh))
    assert "validates nothing about any live agent" in text and "can NOT later be described as untouched holdouts" in text
    r = json.loads((reh / "seed_1" / "baseline_eval.json").read_text(encoding="utf-8"))
    o = r["policy_primary"]["overall"]
    assert f"{o['n_scored']} scored channel-reviews: ideal {o['ideal']}, acceptable {o['acceptable']}, forbidden {o['forbidden']}" in text
    assert "17/17 generator calibration intents" in text and "No scenario was flagged" in text
    assert HR.render({}).endswith("No held-out results exist yet.\n")


# ---------------------------------------------------------------------------------------------- the stored held-out results (skipped before the run)

HELD = ROOT / "evals" / "results" / "heldout"


def stored():
    if not (HELD / "RUN_LOCK.json").exists():
        pytest.skip("the held-out baseline has not been run")
    return {s: (json.loads((HELD / f"seed_{s}" / "provenance.json").read_text(encoding="utf-8")),
                json.loads((HELD / f"seed_{s}" / "baseline_eval.json").read_text(encoding="utf-8"))) for s in GEN["heldout_seeds"]}


def test_each_held_out_seed_has_its_own_complete_provenance_and_results():
    man = F.load_manifest()
    for s, (prov, res) in stored().items():
        assert prov["seed"] == res["seed"] == s and prov["freeze_id"] == man["freeze_id"] and prov["rehearsal"] is False and prov["completed"] is True
        assert {n: v["status"] for n, v in prov["steps"].items()} == {n: "completed" for n in ("generate", "backtest", "tie_checks", "evaluate")}
        assert all(f"seed_{s}" in p for k, p in prov["paths"].items() if k not in ())
        assert res["artifact_hashes"] == prov["hashes"] and prov["hashes"]["decision_db"] == prov["hashes"]["decision_db_at_backtest_start"]
        assert re.fullmatch(r"[0-9a-f]{40}", prov["git_commit"]), "the run records the commit of the original development history it ran at"


def test_held_out_artifacts_differ_across_seeds_and_from_the_development_seed():
    data = stored()
    hashes = [(k, v["hashes"][k]) for _, (v, _r) in data.items() for k in ("decision_db", "hidden_truth", "forecast_store")]
    dev = F.load_manifest()["development_artifacts_seed_1"]
    assert len({h for _, h in hashes}) == len(hashes), "no two held-out artifacts are identical"
    assert not ({h for _, h in hashes} & set(dev.values())), "no held-out artifact equals a development artifact"
    # the forecast store and decision data of a seed were hashed at the SAME time they were built, and every record's seed field agrees
    for s, (prov, res) in data.items():
        truth = ROOT / "data" / "hidden" / f"seed_{s}" / "hidden_truth.json"  # the held-out data is not published; check it only where it exists locally
        if truth.exists():
            assert json.loads(truth.read_text(encoding="utf-8"))["seed"] == s


def test_held_out_results_use_the_registered_definitions_and_state_their_boundary():
    for s, (prov, res) in stored().items():
        assert "says nothing about any live agent" in res["label"]
        assert res["policy_primary"]["overall"]["n_scored"] == 48 and res["policy_primary"]["n_channel_reviews"] == 70
        v = res["scenario_validity"]
        assert v["n_intents"] == 17 and v["n_pass"] + sum(1 for r in v["table"] if not r["pass"]) == 17
        assert bool(v["scenarios_not_materialized"]) == (res["policy_secondary_excluding_unmaterialized_scenarios"] is not None)
        f = res["forecasting"]
        assert f["coverage_nominal"] == 0.80 and f["independent_width_check"]["mean_matches_scorer"] and f["independent_width_check"]["median_matches_scorer"]
        assert len(res["review_log"]) == 70


def test_the_report_and_the_readme_quote_the_stored_results():
    from scripts import build_heldout_report as HR
    text = (ROOT / "reports" / "heldout_baseline_report.md").read_text(encoding="utf-8") if (ROOT / "reports" / "heldout_baseline_report.md").exists() else ""
    if not (HELD / "RUN_LOCK.json").exists():
        pytest.skip("the held-out baseline has not been run")
    assert text == HR.render(HR.load(HELD)), "reports/heldout_baseline_report.md is stale: run scripts/build_heldout_report.py"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for s, (prov, res) in stored().items():
        o = res["policy_primary"]["overall"]
        assert f"{o['n_scored']} scored channel-reviews: ideal {o['ideal']}, acceptable {o['acceptable']}, forbidden {o['forbidden']}" in text
        c = next(x for x in res["forecasting"]["table"] if x["level"] == "cohort" and x["group"] == "overall")
        assert f"| {s} | {c['n']} | {c['wape']:.3f} |" in readme
