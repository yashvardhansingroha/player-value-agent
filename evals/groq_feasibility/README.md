# Groq token-budget feasibility (offline assessment and recorded live updates)

ALL DATA IS SYNTHETIC. This is a chronological log: the UPDATE sections (newest first) record the live Groq runs; the assessment sections further down were written offline, before any Groq call. Token counts from the offline measurements are ESTIMATES. The numbers below are copied from the JSON files
in this folder, which the scripts produced (`measure_tokens.py` -> `measure_tokens.json`, `budget_model.py` -> `budget_model.json`, `feasibility_targeted.py` -> `feasibility_targeted.json`;
run each with the shared venv's `python -I`, the first piped into the second).

## UPDATE 6: the first live runs of the two-stage flow (`reports/phase5_runs/live_groq_2025-11-10_4.json` and `_5.json`): one validated proposal, one provider fault
The protocol allowed ONE live attempt; two were run. Both are preserved untouched and reported here separately. Same case (2025-11-10, dev seed), same model, `reasoning_effort: low`, mode `live`.

**Attempt 4: a validated live agent proposal (status `agent_proposal`, 0 repairs).**

| stage | result |
|---|---|
| investigation | 8 requests, 6 tool calls by the model (overview, then the five policy decisions, one per turn). The model's first closing reply was EMPTY while unity had not been queried, so the required-evidence check fired once (1 nudge); the model then called `get_policy_decision(unity)` and ended with `DONEDONE`. Required evidence complete: yes. Tokens 20,215 prompt + 344 completion. |
| schema | the provider accepted the strict schema (anyOf-with-null unions, `description` keys, container nesting of 5) and `reasoning_effort: low`, and returned text: `finish_reason: stop`, valid JSON, 1,103 completion tokens (cap 2,600). |
| semantic validation | the unchanged validator accepted the answer on the first try: 0 rejections, 0 repairs. Finalization tokens 2,746 prompt + 1,103 completion. |
| total | 22,961 prompt + 1,447 completion tokens; `stages`, `rejections` and the request log are in the file. No replay recording is made for staged runs. |

**What the validated proposal is, and is not.** It is identical to the deterministic rules-only reference on all five channels (applovin, meta, unity: hold; google_uac: insufficient_evidence; tiktok: decrease 85 -> 76.5). It carries 5 summary claims and 11 item claims, and its
reasons are one-line paraphrases of the policy decisions. The model called ONLY `get_review_overview` and `get_policy_decision`: no quality check, economics or experiment tool. So it shows that the staged protocol, structured finalization and the validator work end to end on a real model
for ONE case; it does not show that the model adds anything the rules-only policy lacks. It demonstrates one successful case, not general reliability.

**Attempt 5: fallback in the investigation stage (`rules_only_fallback`, reason `api_failure`).** Six requests and six tool calls (all required evidence retrieved), then request 7 was rejected by the provider with HTTP 400 `tool_use_failed`:
"attempted to call tool 'JSON' which was not in request.tools". The model had ignored the `DONE` instruction and written its full answer as a call to `JSON` (its visible text again used `result.channels...` paths and population `default`, the mistakes the validator rejected before).
In a staged run stage-1 failures are deliberately not unwrapped stage-1 failures, so the run took the labelled fallback and the structured finalization never ran. Nothing is known about how that answer would have fared. Tokens 14,257 prompt + 432 completion, plus the failed request's charge.

**Tally, stated plainly:** staged flow 1 validated proposal, 1 fallback (two attempts). All five full-case live attempts on this one case: 1 validated proposal. Token use today by this log's own accounting is roughly 105,000 to 110,000 of 200,000 (the provider's dashboard is authoritative).

**Now verified live (for this model, on one finalization):** the provider accepts the strict schema as derived, together with `reasoning_effort: low`. Still unverified: how often it does, whether the 429/413 message wording matches the classifier, and the `tool_use_failed` rate in the investigation stage (1 of 2 staged runs).

**Observation, no change made:** attempt 5 failed at the model's "I am done" turn with the required evidence already complete. The stage-1 text is discarded anyway, so treating that specific provider fault as "investigation over" and proceeding to the (still schema-constrained and fully validated) finalization would have been possible. That would be a failure-handling
change made after seeing this failure; it is left as an option for the maintainer and is not implemented.

## UPDATE 5: two-stage design with strict structured finalization (offline only; no live call has been made with it)
Implements option (a). `main`, the frozen baseline and all three failed attempts are untouched. Attempt 3's last answer had `finish_reason: stop` (not `length`); it was degenerate output, not token truncation.

**Stage 1, investigation (unchanged mechanics):** the same bounded local tool loop and evidence log. The provider note now tells the model not to write the answer here and to reply `DONE` when it has the evidence. The model's closing text is
**not** an answer and is not validated. A deterministic check then requires the evidence: `get_policy_decision` for every paid channel, and `plan_experiment` at the policy's own budget for every channel whose RETRIEVED policy decision
is `increase_test`. Failed calls do not count. If something is missing the model is asked (at most `max_evidence_nudges` = 2 times, inside the tool budget) to call it; otherwise the labelled fallback `required_evidence_missing`.
The harness never makes a call on the model's behalf and never waives a requirement.

**Stage 2, structured finalization:** ONE separate non-streaming request with no `tools`, `tool_choice` or `parallel_tool_calls`, and `response_format` = `json_schema`, `strict: true`. The schema is derived from the existing pydantic output
contract by `src/agent/strict_schema.py` (every property required, `additionalProperties: false`, optional fields as `anyOf` with null, `$ref`s inlined, only supported keywords, numeric `new_budget`, the action and qualification enums unchanged;
`check_strict` lists any violation of the restrictions, and the container nesting is exactly 5). The request carries a compact **authoritative evidence index** (`src/agent/evidence_index.py`) built from the validator's own evidence log:
call id, tool, channel, every citable scalar field path with its exact value, grouped under its population label in brackets; failed calls are listed as errors, not evidence. The result goes through the **unchanged** validator
(`validator.py`, `tools.py`, `context.py` and the constraints are byte-identical to the frozen baseline, pinned by a test). One bounded repair is allowed (a second tool-free structured request with the validator's errors); then the labelled fallback.

**Failure handling in stage 2, each explicit and labelled:** provider schema failure (`json_validate_failed`), rejected request, daily/rate limits, timeout, malformed response, model refusal, `finish_reason=length` (incomplete output, not used),
a tool call returned although none was declared, empty text, text that is not JSON. Provider failures are not retried or repaired; they take `finalization_failed`. Non-JSON text goes through the validator path (`malformed_output`, one repair).
The single-stage "unwrap" is not used in a staged run.

**Reporting by stage:** each run record has `stages.investigation` (ended_by, whether the required evidence was complete, nudges, model requests, tokens) and `stages.finalization` (requests, whether the provider accepted the schema and returned
text, whether the text parsed as a JSON object, whether the validator accepted it, repairs, tokens), plus `rejections` (every rejected answer with its error codes). A staged run is not saved as a replay recording (replay does not support two stages yet).

**Offline checks:** 112 tests in the two Groq files; 420 passed, 1 skipped in the wider agent set; 16 mutations of the new safeguards (tools in the finalization request, streaming, `strict` off, missing `additionalProperties`,
required evidence waived, failed calls counted as evidence, validator skipped, unbounded repairs or nudges, refusal or `length` ignored, tokens not counted, population label dropped from the index...) were all caught.
A dry run on the real dev contexts (`staged_dryrun.py`, faked transport with the real tools, index, schema, validator and 8,000-token pacing; targeted scripted investigation) validates for both review dates:
finalization is a 2,800 to 3,100-token request with the full 2,600-token completion cap; with the model's observed one-call-per-turn pattern a run is about 24,000 to 28,000 tokens and about 2 to 3 minutes of pacing, batched about 9,000 to 10,000.

**Unverified until the live attempt (accepted in attempt 4, see UPDATE 6):** whether the provider accepts this schema (anyOf-with-null unions, the `description` keys, container nesting of 5), whether `reasoning_effort: low` with `json_schema` works for this model, and the real size of the index. A rejected schema is an explicit, labelled outcome.
Settings deliberately NOT changed with this stage: `reasoning_effort` (low), the model, temperature, no examples, no weakening of validation.

## UPDATE 4: the third full live attempt (`reports/phase5_runs/live_groq_2025-11-10_3.json`): the API worked end to end, the model's answers did not validate
**Not agent evidence; zero validated agent proposals so far in three attempts.** Offline re-validation of the two recorded answers (`revalidate_run.py`, output in `revalidation_live_groq_2025-11-10_3.txt`; no model call) shows why:

| step | what happened |
|---|---|
| requests | 8: overview, google_uac, tiktok (one call per turn), then request 4 failed with `tool_use_failed` again (the final answer sent as a tool call) |
| unwrap | the answer was taken from `failed_generation` unchanged, labelled `PROVIDER FORMAT RECOVERY`, and given to the validator. **The unwrap path worked as designed on a real failure.** |
| first answer | parsed, then **rejected with 14 errors, all of them correct**: it cited the overview (`tc_001`) with population `default` where the label is `channel_review_status`, and it cited no `get_policy_decision` for applovin, meta or unity (it had not called them). Rule 4 of the system prompt (call it for every paid channel) was not followed. |
| repair | the repair message worked in the sense that the model then called the three missing policy decisions (one per turn) |
| final answer | 724 characters of degenerate output: one item (applovin), `new_budget` set to a string of non-breaking spaces and newlines, raw newline characters inside a JSON string, a stray `"etc":""` key. Invalid JSON, `finish_reason: stop`, 186 completion tokens. Not repairable: the fallback (`malformed_output`) was correct. |
| cost | 21,314 prompt + 543 completion tokens over the 7 successful requests (plus the failed request's charge); about 4 minutes of pacing (waits up to 30 s per request because each request reserves its completion cap) |

What the three attempts show, and do not show: the harness, pacing, id matching, evidence recording, repair loop, validator and fallback behave as designed against the real service. The blocker is the MODEL'S output reliability in this configuration
(gpt-oss-120b via Groq, `reasoning_effort: low`, free-form JSON answer after tool calls): wrapped or degenerate final answers, calls skipped before answering, wrong population labels. Nothing here says a stronger configuration cannot do the task.

Token use today by this log's own accounting: about 65,000 of the 200,000-token daily limit (smoke test plus three attempts including failed requests; the provider's dashboard is authoritative).
`AgentRun.rejections` now records EVERY answer the validator rejects (model turn and error codes), because this run's first rejection had to be reconstructed offline.

**Options, none implemented, each a decision for the maintainer:** (a) constrain the final answer: a final request WITHOUT tools using structured outputs (a JSON schema), if the provider supports that for this model (not checked: the documentation was unreachable);
(b) `reasoning_effort: medium` for the final turn, at a token cost; (c) put one short example answer in the provider note (about 400 tokens, and an example can bias the content); (d) a different model or provider; (e) stop live attempts and keep the rules-only path, the mock plumbing and these three recorded attempts as the honest state.

## UPDATE 3: the second full live attempt (`reports/phase5_runs/live_groq_2025-11-10_2.json`) also ended in the rules-only fallback
**Not agent evidence.** Same case, same model, mode `live`, after the fixes of UPDATE 2 (token-bucket pacing, turn bound 20, provider note asking for batching, two retries of `tool_use_failed`).

| what happened | detail |
|---|---|
| tool calls | again ONE per turn: overview, then policy decisions for applovin, google_uac, meta, tiktok (unity was never queried). The note's request to batch was ignored. |
| pacing | worked as intended: waits of 0, 0, 9, 13 and 15 s (attempt 1 waited about 59 s each). |
| failure | request 6 failed with HTTP 400 `tool_use_failed`: **"attempted to call tool 'JSON' which was not in request.tools"**. `failed_generation` begins `{"name": "JSON", "arguments": {"summary": "The review shows holds for applovin, meta and unity, a decrease for tiktok..."`: **the model was writing its FINAL ANSWER, and the provider parsed it as a function call named JSON.** |
| retries | the identical request failed identically on all 3 attempts (temperature 0): the retry bought nothing and cost roughly 9,000 tokens. |
| tokens | 11,006 prompt + 229 completion over the 5 good requests, plus the three failed attempts. |

What this changes in the picture of the first failure: its `commentary` call came after three policy decisions and may well have been the same fault (an answer or a preamble emitted outside plain message content); the first
run kept only 700 characters of the error, so that cannot be confirmed. The record now keeps up to 6,000.

Two things the saved text also shows about the answer itself (the excerpt is cut at 700 characters, so the rest is unseen): its citation paths begin `result.channels...`, but a field path is relative to the result
(`channels...`), and it stated a decision for unity that no policy-decision call had been made for. The validator would very likely have rejected it; that was never tested because the answer never arrived as text.

**Changes made after seeing this failure (disclosed; the run files are untouched):**
* `max_tool_use_failed_retries` 2 -> 0 (identical retries are futile at temperature 0).
* The provider note now says the final answer is ordinary message text, never a tool call.
* **Unwrapping:** if the provider rejects a generation as a call to a name that is NOT one of our tools, and the `failed_generation` parses as `{"name": <not a tool>, "arguments": {...}}` whose arguments hold the answer's two
  top-level keys (`summary` and `items`), the arguments are used as the model's text answer UNCHANGED. Nothing is repaired, completed or guessed; anything else (a real tool name, other keys, truncated text) still falls back.
  The validator checks it like any other answer, and the run's labels carry `PROVIDER FORMAT RECOVERY: N model answer(s) ...`. This is a deliberate, narrow accommodation of a provider parsing fault, and it can be switched off with
  `unwrap_wrapped_final_answer: false`. Whether it is acceptable for an intended evaluation is a decision for the maintainer.
* The Groq status file now keeps earlier attempts in `history`, as the Gemini one does. (The first attempt's entry had been overwritten; it was restored from the committed copy and marked as reconstructed.)

Cost reality check: with the model making one call per turn, a full run is about 31,000 to 35,000 tokens (six or more requests re-send the growing history), about 6 runs a day on the 200,000-token limit, and three failed attempts at 3,000 tokens each
are not free: the two attempts so far used roughly 50,000 tokens between them.

## UPDATE 2: the first full live case (2025-11-10; `reports/phase5_runs/live_groq_2025-11-10.json`) ended in the rules-only fallback
**Not agent evidence.** Provider groq, model `openai/gpt-oss-120b`, mode `live`. The investigation was valid for five requests and the sixth was rejected by the provider:

| what happened | detail |
|---|---|
| tool calls | ONE per turn, five turns: `get_review_overview`, `get_quality_checks(tiktok)`, `get_policy_decision` for applovin, google_uac, meta. The note said several calls may share a turn; the model did not do that. |
| tokens | 5 successful requests: 11,830 prompt + 245 completion (prompts 1,674 / 2,056 / 2,533 / 2,713 / 2,854). Requests per day left: 994. |
| failure | request 6: **HTTP 400 `tool_use_failed`: "attempted to call tool 'commentary' which was not in request.tools"**. The model emitted a call to a pseudo-tool named after a gpt-oss output channel, and the provider refused to validate it. The bucket header says that failed request still consumed about 3,600 tokens. |
| outcome | `rules_only_fallback`, reason `api_failure`, labelled in the run record; the rules-only proposal passed the same validator. No agent proposal was produced and none was validated. |

**Defects found in the record (fixed afterwards, with the original run kept untouched):**
1. **The pacer waited about 59 s before each of the last four requests; none was needed.** A sliding 60-second window had been modelled. The headers show a continuously refilling token bucket: in every recorded response
   `reset-tokens x (8000/60)` equals `8000 - remaining-tokens` to within 3 tokens (`test_the_bucket_model_reproduces_every_rate_limit_header_the_live_service_returned`). The pacer is now a token bucket whose level is replaced by the provider's
   `remaining-tokens` after every response (also after an error response).
2. **A claim made after the smoke test was only partly right.** The claim was that the provider's accounting ran "a few hundred tokens above usage". The bucket used 1,699 for a request whose usage was 1,701, and 2,608 for 2,578 (the first
   smoke request was +508, one later request +349; reasons unknown, possibly other use of the account). Accounting matches usage on most responses; a margin of a few hundred tokens is kept (500).
3. **`max_model_turns: 10` was too small for what the model actually does.** One call per turn means a 9-call investigation and its answer fill ten turns exactly. Raised to 20 (hard bound; reason recorded in the config).
4. **The adapter treated `tool_use_failed` as a permanent client error.** It is a documented provider response to a malformed model generation, so the same request is now retried at most twice (`max_tool_use_failed_retries`), then the labelled fallback.
   Whether a retry helps is unknown (at temperature 0 the model may fail the same way); this change was made AFTER seeing the failure and is disclosed as such. The provider's message is kept in `error_excerpt`.

**Cost of a single-call-per-turn pattern (`feasibility_targeted.json`, measured ratios 4.3 to 4.9):** ten requests, about 31,000 to 35,000 tokens and 3 to 3.5 minutes per run (the history is re-sent every turn), so about 6 runs per day
against the 200,000-token limit. The same investigation in three batched turns costs about 13,000 to 14,000. The provider note therefore now asks explicitly for the five independent policy-decision calls in ONE turn
(a change to the provider note, recorded as `provider_note_sha256` in every request record). Whether the model will batch, and whether batching makes the `commentary` fault more or less likely, is unknown.

## UPDATE after the first live smoke test (`reports/groq_smoke_status.json`, run by the account holder)
The offline estimates below used a pessimistic token ratio. The smoke test (2 real requests, mode `live`, model `openai/gpt-oss-120b`, 2025-10-13 dev context) measured the real figures:

| measurement | value |
|---|---|
| request 1 prompt (system prompt + provider note + 7 tool schemas + task prompt) | **1,674 tokens** (the offline estimate was 2,981) |
| request 2 prompt (the same plus the overview result and a notice) | 2,073 tokens, so the compact overview result and its messages cost 399 tokens |
| completion tokens | 27 and 23 (reasoning effort `low` was accepted and kept them tiny) |
| chars per token on the serialised request | 4.99 and 4.86; about 4.3 on the tool-result JSON alone (derived: 1,733 added chars for 399 added tokens) |
| rate-limit headers actually returned | `x-ratelimit-limit-requests` 1000, `-limit-tokens` 8000, `-remaining-requests`, `-remaining-tokens`, `-reset-requests` (e.g. 1m26.4s), `-reset-tokens` (e.g. 16.567s): the names the adapter reads were right |
| provider accounting | `remaining-tokens` fell from 8000 to 5,791 after request 1 (2,209) although its usage was 1,701; after request 2 it was 3,801 (4,199 over two requests against 3,797 of usage). SEE UPDATE 2: later responses matched usage closely (1,699 for 1,701), so this was not a systematic overcount |
| first-call behaviour | the model asked for ONE tool (the overview) in its first turn, with provider id `fc_...`; the id was matched to its result; the second reply was valid JSON text |

What this does and does not show: **the adapter can talk to the real service (auth, model id, parameters, one tool call, one matched result, headers parsed).** It is one tiny exchange. It is not an agent run, no proposal was produced or validated,
and nothing about the model's investigation quality is known. The unresolved question is whether the per-minute limit pre-reserves `max_completion_tokens`: the headers are reported after completion, so this test cannot tell, and the pacer
still assumes it does.

**Revised feasibility with the measured ratios (4.3 on result JSON, 4.9 on the prefix; `budget_model.json`, `feasibility_targeted.json`):**
* Fixed prefix about 1,700 tokens, not 2,200 to 3,200: about 5,700 tokens of each minute are free for history and answer.
* A TARGETED investigation ends with a final-request prompt of about 3,900 to 5,100 tokens and the full 2,600-token completion cap fits; about 12,700 to 20,000 tokens per run, 9 to 12 runs per day by the 200,000-token limit.
* A FULL SWEEP is still too large: even the compact view ends at about 7,200 to 8,800 prompt tokens, above the roughly 6,200 that leave 1,200 for the answer. The preflight refuses it unsent and the run takes the labelled rules-only fallback.
* Pacing still imposes about 2 minutes per run because every request reserves its completion cap; that is the conservative reading and costs time, not correctness.

The 'Recommendation' at the bottom of this file was written BEFORE the smoke test and is superseded: request 1 came in at 1,674 tokens, far under its 2,600 gate, so the gate says a single full case is now worth trying.

## Account limits (as stated by the account holder, candidate `openai/gpt-oss-120b`)
30 requests/min, 1,000 requests/day, **8,000 tokens/min**, 200,000 tokens/day. A request that is by itself larger than the per-minute token limit can never succeed, however long we wait.

## Counting method and its uncertainty
* **Characters are measured exactly**, from the real serialised text of the dev-seed reviews 2025-10-13 and 2025-11-10: system prompt 4,090 chars; the seven tool declarations in OpenAI function format
  2,801 chars; task prompt 502 chars; each tool result as the model would see it (full envelope vs the compact view the adapter sends).
* **Tokens are estimated as chars / ratio.** No tokenizer for gpt-oss is available offline (and fetching one would be a network call), so three ratios bracket the truth: 2.6 (pessimistic, number-heavy
  JSON), 3.2 (central), 3.8 (optimistic). Treat the central figure as plus or minus 20 percent. The chat template adds markers per message and re-renders tool schemas; the model adds 12 tokens per message and
  15 percent to the tool schemas (unverified). Reasoning tokens of a gpt-oss model count as OUTPUT; the model assumes 250 to 450 tokens for tool-calling turns and 1,400 for the final answer.
* **The first live response replaces the estimate**: the adapter logs the provider's `usage.prompt_tokens`, recomputes chars/token for the same serialisation and uses it (5 percent safer) for every later request.
* Not verified against the provider: whether the per-minute limit pre-counts the requested `max_completion_tokens`. The adapter assumes it does (the conservative reading).

## What the measurement says
| item | chars | tokens at 2.6 / 3.2 / 3.8 |
|---|---|---|
| system prompt + task prompt | 4,592 | 1,766 / 1,435 / 1,208 |
| seven tool declarations (x1.15 template allowance) | 2,801 | about 1,240 / 1,000 / 840 |
| one tool result, compact view: overview / policy decision / quality checks / forecast / economics / plan / constraints | about 1,100 / 470-660 / 1,800-1,900 / 1,200-1,800 / 1,000-1,800 / 1,500 / 335 | divide by the ratio |

The compact view (no whitespace; `output_sha256`, `args`, and the repeated `as_of` / `policy_hash` dropped after the first result) saves 15 to 21 percent on the large results and 58 percent on the constraint check.
The authoritative envelopes, population labels, citations and validator are unchanged: only the text SENT to the model is compacted.

**The fixed prefix alone is about 2,200 to 3,200 tokens**, so only about 4,800 to 5,800 tokens of an 8,000-token minute are left for history and answer.

**Whole-run model (`budget_model.json`, every request re-sends the history):**
* The investigation Gemini actually performed (overview, then four per-channel tools for all five channels, 21+ calls): the request that carries all results is **9,200 to 16,600 estimated tokens in every
  scenario, compact view or not. It does not fit one minute and can never be sent.** Waiting between requests cannot fix this.
* A TARGETED investigation (overview, the five policy decisions, then quality checks only for decreases and abstentions, economics only for decreases, the plan only for an increase) ends with a final-request
  prompt of about 4,800 (optimistic) / 5,700 (central) / 6,900 (pessimistic) tokens, about 20,000 to 28,000 tokens per run.

**Through the real adapter code on the real dev contexts (`feasibility_targeted.json`; a faked transport reports usage at the stated ratio; the fake 'model' asks for 1 + 5 + 3 calls in three turns and then
sends a placeholder, so a `malformed_output` fallback in that file is expected and means nothing):**

| ratio | final-request prompt (2025-10-13 / 2025-11-10) | completion room left in the minute | total tokens | time imposed by the 8,000/min pacing |
|---|---|---|---|---|
| 3.8 | 5,059 / 5,140 | 2,374 / 2,289 | about 15,800 | about 180 s |
| 3.2 | 6,008 / 6,104 | 1,376 / 1,275 | about 18,300 | about 180 s |
| 2.6 | refused before sending (7,784 / 7,908 estimated) | none | n/a | n/a |

## Conclusions
1. A full-sweep investigation is **infeasible** on this account. The adapter keeps the complete evidence for validation and shows a compact view, but does not drop results the model must cite.
2. A **targeted** investigation is **feasible only if the real chars-per-token is about 3.2 or better**, and then the final answer has only about 1,300 tokens of room (the answer is estimated at 1,100 to 1,500
   plus reasoning), so it is tight. At 2.6 it fails. The estimate is too uncertain to decide offline.
3. A run takes about 3 minutes (the pacer reserves the completion cap for every request) and about 16,000 to 18,000 tokens, so roughly 10 runs per day fit the 200,000-token daily limit; requests per minute and
   per day are not the constraint.
4. The investigation is only as targeted as the model chooses to be. A model that sweeps every tool takes the labelled rules-only fallback (`REQUEST TOO LARGE (preflight)`); no result is silently dropped.

## Recommendation (written before the smoke test; superseded by the update above)
**Proceed, but only to the small smoke test, and gate everything after it on its measurement.** `scripts/groq_smoke.py --live` sends the real system prompt, provider note, all seven tool declarations and the
real task prompt (about 3,000 estimated tokens), then one tool round trip, and prints the provider's actual `prompt_tokens` for request 1.
* Request-1 `prompt_tokens` **at or below about 2,600** (chars/token about 3.2 or better): a full single-case run is worth trying.
* **Between 2,600 and 3,200**: borderline; the final answer may be truncated. Do not run a full case until the prompt or the tool schemas are shortened (a deliberate change to frozen agent text, to be recorded).
* **Above about 3,200** (chars/token 2.6 or worse): STOP. A full run cannot fit with these prompts and tools.
Options if it stops (none implemented, each a decision for review): shorten the tool descriptions; drop `get_forecast_and_populations` / `check_budget_constraints` for this provider (the validator re-derives
constraints itself); raise the plan to a tier with a higher per-minute limit.
