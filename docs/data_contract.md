# Data contract (v1)

ALL DATA IS SYNTHETIC.

## What the database supports

- **Immutable facts that arrive late.** A player, a player-day or a spend row is written once. It becomes visible from 00:00 of
  its `reported_at` date. An earlier review (an earlier `as_of`) never sees it; a later review does. Late arrival is inserted,
  never overwritten.
- **Declared completeness.** `cohort_reporting.complete_at` is the date from which every fact for install date .. install date +
  milestone is final. A late fact must arrive no later than that date (a test checks the generated data obeys this).
- **Recomputation from the database.** Metrics are recomputed from the base tables through the `v_*` availability views on every
  call. There is no cache of historical outputs.

## What it does NOT support

- **Overwriting or deleting a fact that was already visible at an earlier `as_of`.** The schema keeps no record versions and no
  historical snapshots. After such a correction, recomputing an earlier date from the current tables gives a different answer
  than the review saw. v1 does not try to handle this. Supporting corrections would require retained record versions (for
  example valid-from / valid-to columns per fact) or stored historical snapshots.
- Anything the tests cannot show. The historical-invariance tests cover insertion/alteration of **not-yet-available** records;
  separate positive-control tests show that editing **already-visible** records changes recomputed metrics. Together they
  document the boundary. They are selected tests and do not prove that history is correction-safe.

## Consequence for later phases

Phase 2 will store each issued forecast's inputs and outputs. That preserves what was forecast, but it does **not** make the
underlying database able to reconstruct history after a correction. A forecast that must be re-derived after a correction cannot
be, in v1; the stored record is the only faithful copy of what the review saw.

## Calibration proxy

`true_step20_marginal_roas_over_target` (generator QA) is an **approximate calibration proxy**: it assumes value per install is
proportional to player quality. It is not exact simulation truth and must never be used as the oracle for scoring future
experiments; Phase 4 scores experiments with the experiment simulator's own truth.
