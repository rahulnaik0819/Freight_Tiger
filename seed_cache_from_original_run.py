"""One-off: rebuild the LLM verdict cache from the ORIGINAL notebook run.

The original run's checkpoint (spike_results.json) was not kept, but its verdicts
survive in the originally submitted CSV (outputs/original_submission.csv). Every row
whose reason is not the rule-based 'No note matches...' text came from a live
gemini-3.1-flash-lite call (the notebook's own summary shows 21 'ok' rows, all from that
model). This script re-creates those 21 cache entries so the new pipeline reproduces
the original verdicts without new API spend.

Token usage for these entries is unknown (the original run did not record it), so it is
stored as null and excluded from token totals.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import freight_audit as fa

orig = pd.read_csv(fa.OUT_DIR / "original_submission.csv", keep_default_na=False)
orig["week_of"] = pd.to_datetime(orig["week_of"])
weekly = fa.build_weekly_table(fa.load_shipments(fa.DATA_DIR / "shipment_records.csv"))
spikes = weekly[weekly["is_spike"]].set_index(["route", "week_of"])
retriever = fa.Retriever(fa.load_notes(fa.DATA_DIR / "context_notes.csv"), "tfidf")

cache, skipped = {}, 0
for r in orig.itertuples():
    spike = spikes.loc[(r.route, r.week_of)].copy()
    spike["route"], spike["week_of"] = r.route, r.week_of
    cands = retriever.candidates(r.route, r.week_of)
    if cands.empty:
        skipped += 1
        continue
    cache[fa.LLMJudge.cache_key(spike, cands)] = {
        "verdict": {"flagged": r.flagged, "matched_note_id": r.matched_note_id, "reason": r.reason},
        "model": "gemini-3.1-flash-lite",
        "usage": {"input_tokens": None, "output_tokens": None},
        "source": "original_run",
    }
path = fa.OUT_DIR / "llm_cache.json"
path.write_text(json.dumps(cache, indent=1, sort_keys=True))
print(f"seeded {len(cache)} verdicts from the original run; {skipped} rule rows skipped -> {path}")
