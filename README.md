# FreightTiger – Freight Cost Anomaly Auditor

Watches weekly cost per tonne-km per route, flags weeks that look out of the ordinary, and explains each
flag from the provided context notes – or says "unexplained" when no note genuinely supports it.

```
python freight_audit.py run            # writes outputs/final_submission.csv (+ audit_detail.csv, run_log.json)
python freight_audit.py run --no-cache # force live Gemini calls (needs GEMINI_API_KEY env var or Colab secret)
python freight_audit.py validate       # contract + grounding check on the CSV
python eval_harness.py                 # 28 tests + gold-set scoring -> outputs/eval_report.json
python repro_check.py [--live --model M]  # 3 runs, diff -> outputs/repro_report.txt
```
Colab: upload the repo, set the `GEMINI_API_KEY` secret, then `!python freight_audit.py run --no-cache`.
`original_notebook.ipynb` is the first version, kept for reference.

## How it works

| Stage | What it does | Deterministic? |
|---|---|---|
| 1. Weekly table | Group by route + `route_type` + Monday-start week; cost per tonne-km = Σ cost / Σ(tonnes × km) | yes |
| 2. Baselines | **Own history:** mean of the trailing 8 calendar weeks *strictly before* this week. **Peers:** mean of every other route with the same `route_type` that week (self excluded) | yes |
| 3. Flag | cost > 1.10 × history **or** > 1.10 × peers | yes |
| 4. Retrieval | Hard filter: note applies to the route (or "All Routes") and is dated within 21 days before the week through the week's end. Survivors ranked by similarity (MiniLM; TF-IDF fallback), top 3 | yes |
| 5. LLM judge | Gemini, temperature 0, JSON-schema output, reads only the surviving notes | cached / temp 0 |
| 6. Verifier | A "justified" verdict stands only if the cited note: applies to the route, is not dated after the week, has a stated period that overlaps the week, and describes a cost increase (not "stable", "no rate change", "returned to normal"…) | yes |
| 7. Reason text | Built from the cited note's own text plus computed numbers – never free LLM prose for justified rows | yes |

**No candidate note → no LLM call → "unexplained"** (69 of 90 flags). The LLM can never create a justification the verifier rejects.
If the API fails for a row the run aborts (no `ERROR` rows in the CSV) unless `--fallback verifier` is passed.

### Baseline details the brief asks me to state
- Rows with fewer than 8 prior weeks use all prior weeks available – nothing padded or extrapolated. Warm-up weeks are the first 8 weeks of each route; 5 flagged rows fall there.
- The first week of each route has no history: `vs_own_history` is blank (1 flagged row: Mumbai-Pune 2024-01-01, flagged on the peer comparison).
- `cost_per_tonne_km` is rounded to 2 dp (as in the sample); percentages are computed from unrounded values.

## What was changed from the first version (and why)

| Problem found | Fix |
|---|---|
| No README, repro evidence, token log; eval was 7 assertions that silently skipped missing rows | `README.md`, `repro_check.py`, token accounting in `run_log.json`, `eval_harness.py` |
| Period check lived only in the LLM prompt; code only checked "note not dated after week" | Independent verifier (route, date, period overlap, increase wording); LLM and verifier must agree |
| Justified reasons sometimes omitted the note date (brief: "name the note and reflect its content and dates") | Reasons assembled from note id, date and text |
| Failed API calls produced `ERROR` rows that would break the CSV contract | Run aborts / explicit fallback; incomplete runs write to `*_incomplete` files and never overwrite good output |
| `RESET = True` default deleted the checkpoint on every run; Colab-only secret access | Verdict cache doubles as checkpoint (`--no-cache` is explicit); env var with Colab fallback |
| Model-chain fallback could silently change verdicts between runs | Model recorded per row; verdicts cached by (route, week, note ids + text hash, prompt version) |
| Notebook cells mixed everything; peer average via row-wise `apply` | One module, vectorised peers (verified identical to the original numbers) |
| History window relied on row order (breaks if a week is missing) | Calendar-week window; tested with a deliberately gapped dataset |
| Sample CSV row 4 has an unquoted comma (not strictly valid CSV) | Validator uses only its header as the column contract; our output is properly quoted |
| `retrieval TOP_K` never mattered (max 2 candidates) | Documented honestly below instead of presenting it as meaningful ranking |

Prompt text, JSON schema, temperature, model chain and flag threshold are **unchanged** from the original, so the 21 verdicts from the original run remain valid evidence
(`seed_cache_from_original_run.py` rebuilt them from the original CSV; provenance `original_run` in `outputs/llm_cache.json`).

## How I convinced myself it is right

`python eval_harness.py` → 28 tests pass, report in `outputs/eval_report.json`.

1. **Independent oracle:** a naive pure-python re-implementation (loops and dicts, different algorithm) matches the vectorised pipeline on all 728 route-weeks (cost/tonne-km, both baselines, prior-week counts, flags).
2. **Property tests:** totals conserved; weeks are Mondays; no look-ahead (tripling future costs leaves earlier baselines unchanged); a route's own costs don't change its peer average; missing-week handling; bad input raises.
3. **Tests have teeth:** I injected six bugs (look-ahead in history, self in peers, Sunday week start, period check off, negation guard off, verifier bypassed). Every one made tests fail.
4. **Verifier matrix:** 20 note × route × week cases written from the note texts (wrong route, week before the note, flood already over, festival week over, "not affected" notes…).
5. **Adversarial judge tests (LLM stubbed):** citing an unretrieved note, an expired note, a wrong-route note, a future-dated note, a "no rate change" note, or drifting the label never reaches the CSV. Also malformed JSON, 404, total failure, cache and token accounting.
6. **Contract mutation tests:** corrupted outputs (wrong cite, blank cite, unknown note, `ERROR` value, missing column/row) are all caught.
7. **Gold set** (`eval/gold_labels.csv`, 21 hand-labelled candidate-bearing spikes; the other 69 have no note for their route/window):

| | accuracy | justified precision | justified recall | correct note cited |
|---|---|---|---|---|
| lenient (N003 weeks after the note week = justified) | 21/21 | 1.00 | 1.00 | 21/21 |
| strict (those 3 ambiguous rows = unexplained) | 18/21 | 0.57 | 1.00 | 21/21 |

   The 3 ambiguous rows are Chennai-Bangalore, Mumbai-Delhi and Mumbai-Pune for 2025-05-12: N003 (dated 2025-05-05) says diesel rose "starting this week". I read that as open-ended; a strict reader could say only the week of 05-05 is covered. Data supports persistence (costs stayed elevated for several weeks), but it is a judgement call – see limitations.
8. **Verifier vs LLM:** on the 21 LLM-judged rows they agree 21/21, so the two layers independently reach the same 7 justified / 14 unexplained.

## Reproducibility
Brief: "same input, same output every time". Three kinds of evidence, in order of strength:

1. **Deterministic stages** (weekly numbers, baselines, flags, retrieval, verifier, reason text, ordering, CSV formatting): `python repro_check.py` re-runs the pipeline three times with cached LLM verdicts and produces byte-identical CSVs.
2. **First live attempt (3 runs, `--no-cache`, fallback chain on) – disclosed, not hidden:** numbers and 89 of 90 verdicts were identical across all three runs; **one row flipped in run 3** (Mumbai-Pune 2025-05-12, the borderline diesel-note row: justified in runs 1-2, unexplained in run 3). Cause: Gemini 429/503 errors pushed rows onto different models between runs (run 1: 20 rows on `gemini-3.1-flash-lite` + 1 on `gemini-3.5-flash-lite`; run 2: 21 on `gemini-3.1-flash-lite`; run 3: 12 + 9 on `gemini-3.5-flash-lite`). Runs 1 and 2 agreed on every row. Per-judgment consistency: 62/63.
3. **Fix and re-test:** `--model gemini-3.1-flash-lite` pins one model (no fallback, up to 8 retries with capped backoff), so a verdict cannot depend on which model was available that minute. Re-run with `python repro_check.py --live --model gemini-3.1-flash-lite`; the outcome is in `outputs/repro_report.txt`.

LLM settings: temperature 0, JSON-schema output, verdicts cached per (route, week, note ids/text, prompt version). The verifier means a changed LLM answer can only turn a justification *off*, never invent one that the cited note's route, date, period and wording do not support. The one row that flipped is exactly the case where the verifier passes and the judge may disagree – that is the residual live-LLM risk.

## Cost of one full run (measured)
From `outputs/run_log.json`, one full live run over the whole shipment file (`python freight_audit.py run --no-cache`):

| Metric | Value |
|---|---|
| LLM calls (successful) | 21 (all `gemini-3.1-flash-lite`); the other 69 flagged rows have no candidate note and need no call |
| Failed attempts (429/503, retried) | 7 |
| Total input tokens | 5,364 |
| Total output tokens (incl. thinking) | 1,494 |
| Wall time | ~134 s (dominated by retries/backoff) |
| Cost at published paid rate ($0.25 in / $1.50 out per 1M tokens, Gemini 3.1 Flash-Lite Preview, ai.google.dev/pricing) | ≈ **$0.0036** |
| Cost on the free tier | $0 |

(The pre-run estimate in `outputs/eval_report.json` – ≈4.6k in / 1.2k out – was within ~15% of the measurement.)

## Known limitations (be ready to discuss)
- **Peer baseline is thin.** Short and Long groups have only two routes each, so "peers" is a single route. Mumbai-Pune sits ~12% above Delhi-Jaipur in the median week, so it is flagged chronically: 82 of 90 flags are peer-only, 67 are Mumbai-Pune, and 11 flagged rows are not rising versus their own history. I kept the brief's definition and the OR rule (the sample file's Mumbai-Pune row is exactly such a flag). The 10% threshold is my assumption; sensitivity is in `eval_report.json` (1.05→219 flags, 1.10→90, 1.15→27, 1.20→20).
- **An all-routes note cannot explain a gap to peers.** N003 raised every route, so for flags caused only by the peer comparison the reason carries an explicit caveat instead of claiming a full explanation.
- **Retrieval is a hard filter, not semantic ranking.** With 10 notes the filter leaves at most 2 candidates, so top-k never binds and the embedding model cannot change which notes the judge sees. Similarity-only retrieval would have put a route- or date-invalid note in the top 3 for 90/90 spikes (TF-IDF test), which is why the filter exists.
- **Verifier wording rules are tuned to this note corpus** (regex cues for "increase" / "not affected"); new phrasings would need new cues. The gold set is small and was labelled by the author after seeing outputs.
- **Not done:** the stretch Q&A ("why did this route get pricier in March?").
- **Not executed in the build environment:** the live Gemini call path and the MiniLM embedder (no API key; HuggingFace blocked). Tests used a stubbed LLM and the TF-IDF retriever; the live code uses the same call, schema and prompt as the original.
