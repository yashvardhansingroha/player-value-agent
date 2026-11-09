"""README.md is generated from the result files; it must match a fresh render, say what is pending, and carry no key or unsupported claim."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import build_readme as B  # noqa: E402

README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_the_committed_readme_equals_a_fresh_render_from_the_result_files():
    assert README == B.render(), "README.md is stale: run `python scripts/build_readme.py` after regenerating the results"


def test_result_numbers_in_the_readme_come_from_the_result_files():
    ev = json.loads((ROOT / "evals" / "results" / "dev_eval.json").read_text(encoding="utf-8"))
    o = ev["policy"]["overall"]
    assert f"**{o['ideal']} ideal, {o['acceptable']} acceptable, {o['forbidden']} forbidden, {o['unlisted']} unlisted.**" in README
    f0 = next(r for r in ev["forecasting"]["table"] if r["level"] == "cohort" and r["group"] == "overall")
    assert f"{f0['wape']:.3f}" in README and f"{f0['coverage']:.3f}" in README and f"{f0['interval_width_over_forecast_mean']:.3f}" in README
    cons = ev["constraints"]
    assert f"**{sum(c['refused_as_expected'] for c in cons)} of {len(cons)} refused as expected.**" in README


def test_the_readme_states_what_is_pending_and_what_is_not_shown():
    assert "ALL DATA IS SYNTHETIC" in README and "SANDBOX" in README and "SIMULATED" in README
    assert "Gemini: **PENDING**" in README and "0 valid live-model run(s)" in README and "HTTP 402" in README
    assert "No live run has yet shown the agent working" not in README, "stale claim: a validated live Groq proposal exists (one case)"
    assert "**Gemini produced no valid live run.** The only validated live agent proposal is the single Groq one described next" in README
    assert "no agent-vs-baseline comparison of decision quality exists" in README
    for part in ("(a) Development results", "(b) Held-out deterministic-baseline results", "(c) Mock agent results", "(d) Live agent verification"):
        assert part in README, part  # the four kinds of result are named apart
    state = B.heldout_state()["state"]
    if state == "untouched":
        assert "The held-out seeds are untouched and nothing has been frozen" in README and "## Held-out results" not in README
    elif state == "frozen_not_run":
        assert "not run yet" in README and "still untouched" in README and "## Held-out results" not in README
    else:  # the run happened: the outdated claim must be gone and the boundary stated
        assert "untouched and nothing has been frozen" not in README and "the held-out evaluation has not been run" not in README
        assert "## Held-out results: the frozen deterministic baseline only" in README
        assert "validates only the rules-only baseline" in README and "No model was called" in README and "no longer be called untouched holdouts" in README
        assert "No agent was evaluated on the held-out seeds" in README
    assert "Adequacy for any other channel is unknown" in README and "not independent proof of economic harm" in README
    assert "MOCK" in README and "below nominal" in README


def test_the_readme_has_every_planned_section_and_no_key_like_string():
    for h in ("## What this is", "## Development results", "## Key definitions", "## Assumptions and limitations", "## Run it", "## How the code is checked", "## Extending to LiveOps", "## Repository map"):
        assert h in README, h
    assert not re.search(r"AIza[0-9A-Za-z_\-]{20,}", README) and "GEMINI_API_KEY" in README and "never in a file" in README
