# Provenance of the evaluation records

ALL DATA IS SYNTHETIC. This file says what the committed evaluation records are, what can be re-checked from this repository, and what cannot.

## History

This repository starts from a single commit. The project was developed in a longer local git history that is not published; nothing here pretends otherwise. The original commit ids below are recorded for reference only and **cannot be resolved from this repository**:

| step | original commit | what it recorded |
|---|---|---|
| scenario expectations | `c3447c8` | `evals/scenarios/*.json` expectations, committed before any synthetic data was generated |
| Phase 6 criteria | `c268136` | `evals/criteria.yaml`, committed before the first Phase 6 evaluation run |
| baseline freeze | `abd7625` (annotated tag `baseline-freeze-v1`) | the amended baseline-only protocol (`evals/heldout_protocol.yaml`) and `evals/FREEZE_MANIFEST.json`, committed before any held-out data was generated |

The held-out run was then executed once on seeds 9101, 9102 and 9103 (`evals/results/heldout/RUN_LOCK.json` records the commit it ran at, `abd7625e711f0422c4151c3403c973618db9d38c`). The Groq adapter and the recorded Groq runs were produced later, in a separate branch of that history, and are merged into this single implementation. The test `tests/test_scenarios.py::test_expectations_are_unchanged_since_the_pre_generation_commit` compares against `c3447c8` when that commit exists and is skipped otherwise, which is the case here.

## What can be re-checked here

- **The freeze manifest.** `evals/FREEZE_MANIFEST.json` holds the sha256 (line endings normalised) of the 78 files that were frozen. `python scripts/verify_baseline_provenance.py` compares the current files with it. The annotated tag is not available in this repository, so the tag check is not run; file content is.
- **Declared differences.** Seven files differ from, or were added after, the freeze. They are listed with their current hash and reason in `evals/publication_delta.json`: the Groq adapter (`src/agent/agent.py`, `src/agent/llm.py`, `src/agent/groq_client.py`, `src/agent/strict_schema.py`, `src/agent/evidence_index.py`, `config/agent_groq.yaml`) and one wording edit to `evals/heldout_protocol.yaml`. The system prompt and tool declarations are unchanged (the manifest's agent-surface hashes still match). The held-out baseline results were produced by the frozen files, not by the changed versions. The forecasting, quality, policy, SQL and configuration files that compute the baseline are byte-identical to the freeze (they are not in the declared list); the changed files are in the agent layer. This repository has not re-run the held-out evaluation to demonstrate that the results are unaffected, and it must not (see below).
- **The development data.** `python scripts/generate_data.py` regenerates the dev seed 1 data; its decision database and hidden-truth file hash to the values in the manifest (`development_artifacts_seed_1`). The forecast store hash additionally requires the backtest.
- **Held-out results.** `evals/results/heldout/seed_*/baseline_eval.json` and `provenance.json` hold the results and the sha256 of the data artifacts (decision database, hidden truth, forecast store) each seed was computed from. The held-out data itself is not included.
- **Live-run records.** `reports/groq_runs_manifest.json` holds the sha256 of the original Groq run files; `app/groq_view.py` and the README builder refuse to show numbers if a file does not match.

## What cannot be re-checked here

- The ordering of the original commits above (it needs the original history).
- That the held-out data was generated only after the freeze (the run lock and provenance record the commit and the artifact hashes; the data is not published).
- Anything about the Gemini attempts beyond the recorded files in `reports/phase5_live_status.json` and `reports/phase5_runs/`.

## Rules for the consumed seeds

Seeds 9101-9103 were used by the baseline run. They are not untouched holdouts for any agent evaluation. `scripts/run_heldout_baseline.py` refuses a second run (run lock) and is documented here only as the record of how the held-out run was performed; do not re-run it. Use new seeds for any new held-out evaluation, with its own freeze.

## Edits made when preparing this repository

- Machine-specific absolute paths in `evals/results/**/provenance.json` were rewritten to repository-relative paths. No hash, result or timestamp was changed.
- Text files were normalised to LF line endings (the freeze hashes already normalise line endings).
- The README generator and the screen text were reworded to describe this single implementation. Result numbers are unchanged and still read from the result files.
