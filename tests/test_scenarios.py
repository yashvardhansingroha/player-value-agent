"""Scenario files: structure, agreement with the generator config, and immutability of the expectations
written before any data was generated (compared with the pre-generation git commit)."""
import json
import subprocess
from pathlib import Path

import pytest

from src.generator.generate import load_config
from src.generator.intents import load_scenarios

ROOT = Path(__file__).resolve().parents[1]
ACTIONS = {"increase_test", "hold", "decrease", "insufficient_evidence"}
PRE_GENERATION_COMMIT = "c3447c80fe762825443c3900228d656e8ecaa4ac"  # scenario files + config, before any generator code or data
PLANT_KEYS = {"tracking_outage": "tracking_outage", "cpi_spike": "cpi_spike", "tiny_sample_whale": "tiny_sample_whale"}


def test_scenario_files_are_well_formed():
    cfg = load_config()
    scenarios = load_scenarios()
    assert set(scenarios) == {
        "steady_winner", "front_loaded", "saturating_imperfect_curve", "tracking_outage",
        "cpi_spike", "delayed_reporting", "tiny_sample_whale", "regime_shift",
    }
    for sid, sc in scenarios.items():
        assert sc["id"] == sid
        e = sc["expected"]
        for key in ("ideal_actions", "acceptable_actions", "forbidden_actions"):
            assert set(e[key]) <= ACTIONS
        assert not set(e["ideal_actions"]) & set(e["forbidden_actions"])
        assert not set(e["acceptable_actions"]) & set(e["forbidden_actions"])
        assert set(sc["evaluation_as_of_days"]) <= set(cfg["decision_as_of_days"])
        assert sc["channel"] in cfg["channels"] or sc["channel"] == "all_paid"
        assert sc["calibration_intents"], sid
        for intent in sc["calibration_intents"]:
            assert intent["op"] in (">=", "<=")


def test_decision_dates_are_mondays_and_after_the_burn_in():
    cfg = load_config()
    assert all(d % 7 == 0 for d in cfg["decision_as_of_days"])  # day 0 is a Monday
    assert min(cfg["decision_as_of_days"]) >= 240


def test_scenario_windows_and_channels_match_generator_config():
    cfg, scenarios = load_config(), load_scenarios()
    for sid, key in PLANT_KEYS.items():
        planted = cfg["planted"][key]
        assert scenarios[sid]["planted_window_days"] == [planted["start_day"], planted["end_day"]]
        assert scenarios[sid]["channel"] == planted["channel"]
    assert scenarios["regime_shift"]["planted_window_days"][0] == cfg["planted"]["regime_shift"]["start_day"]
    assert scenarios["delayed_reporting"]["channel"] == "unity"


def test_scenario_evaluation_dates_do_not_overlap_between_scenarios_on_one_channel_unless_intended():
    scenarios = load_scenarios()
    seen = {}
    for sid, sc in scenarios.items():
        if sc["channel"] == "all_paid":
            continue
        for d in sc["evaluation_as_of_days"]:
            seen.setdefault((sc["channel"], d), []).append(sid)
    overlaps = {k: v for k, v in seen.items() if len(v) > 1}
    # applovin and unity each carry one whole-period scenario; google_uac and tiktok split dates between two scenarios
    assert not [k for k in overlaps if k[0] in ("google_uac", "tiktok")], overlaps


def test_expectations_are_unchanged_since_the_pre_generation_commit():
    try:
        subprocess.run(["git", "cat-file", "-e", PRE_GENERATION_COMMIT], cwd=ROOT, check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("pre-generation commit not available in this checkout")
    for sid, current in load_scenarios().items():
        original = json.loads(
            subprocess.run(["git", "show", f"{PRE_GENERATION_COMMIT}:evals/scenarios/{sid}.json"], cwd=ROOT,
                           check=True, capture_output=True, text=True).stdout
        )
        for key in ("channel", "planted_window_days", "evaluation_as_of_days", "expected", "if_experiment_run"):
            assert current[key] == original[key], f"{sid}.{key} changed after the pre-generation commit"
