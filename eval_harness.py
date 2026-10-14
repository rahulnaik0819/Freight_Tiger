"""Evaluation harness: why the submission can be trusted.

Run:  python eval_harness.py          (writes outputs/eval_report.json, exit 1 on any failure)

Layers
  1. Phase-1 oracle      - a naive pure-python re-implementation (different algorithm)
                           must match the vectorised pipeline on all 728 route-weeks.
  2. Property tests      - conservation, no look-ahead, self-excluded peers, gap handling.
  3. Verifier matrix     - note x route x week cases written by hand from the note texts.
  4. Judge guard tests   - adversarial LLM outputs (stubbed, no API) must never reach the CSV.
  5. Contract mutations  - the CSV validator must catch deliberately corrupted outputs.
  6. Gold-set scoring    - eval/gold_labels.csv (hand-labelled) vs outputs/final_submission.csv.
  7. Diagnostics         - retrieval ablation, threshold sensitivity, verifier-vs-LLM agreement.

Limits (also stated in README): the LLM is stubbed in layer 4, so these tests prove the
guardrails, not the judge's accuracy; the gold set is small (21 hand-labelled rows) and was
labelled by the same person who wrote the verifier.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import freight_audit as fa

SHIP = fa.load_shipments(fa.DATA_DIR / "shipment_records.csv")
NOTES = fa.load_notes(fa.DATA_DIR / "context_notes.csv")
WEEKLY = fa.build_weekly_table(SHIP)
RETRIEVER = fa.Retriever(NOTES, "tfidf")  # offline + deterministic
NOTE = NOTES.set_index("note_id", drop=False)
T = pd.Timestamp


# ----------------------------------------------------------------------------
# 1 + 2. Phase 1 oracle and property tests
# ----------------------------------------------------------------------------
def naive_weekly(ship: pd.DataFrame) -> dict:
    cost, tkm, rtype = defaultdict(float), defaultdict(float), {}
    for r in ship.itertuples():
        route = f"{r.origin}-{r.destination}"
        monday = (r.shipment_date - timedelta(days=r.shipment_date.weekday())).date()
        cost[(route, monday)] += r.freight_cost_inr
        tkm[(route, monday)] += r.quantity_tonnes * r.distance_km
        rtype[route] = r.route_type
    cpt = {k: cost[k] / tkm[k] for k in cost}
    out = {}
    for (route, wk), v in cpt.items():
        prior = [cpt[(route, wk - timedelta(weeks=i))] for i in range(1, fa.HIST_WINDOW + 1)
                 if (route, wk - timedelta(weeks=i)) in cpt]
        peers = [cpt[(r2, wk)] for r2 in rtype
                 if r2 != route and rtype[r2] == rtype[route] and (r2, wk) in cpt]
        hist = sum(prior) / len(prior) if prior else float("nan")
        peer = sum(peers) / len(peers) if peers else float("nan")
        out[(route, wk)] = (v, hist, peer, len(prior))
    return out


def compare_to_naive(ship: pd.DataFrame, weekly: pd.DataFrame):
    oracle, bad = naive_weekly(ship), []
    for r in weekly.itertuples():
        v, h, p, n = oracle[(r.route, r.week_of.date())]
        got = (r.cost_per_tonne_km, r.historical_avg_8w, r.peer_avg_this_week)
        if not np.allclose(got, (v, h, p), equal_nan=True, rtol=1e-9) or r.n_prior_weeks != n:
            bad.append((r.route, r.week_of.date()))
        spike = (v > h * fa.THRESHOLD) or (v > p * fa.THRESHOLD)  # NaN compares False
        if bool(r.is_spike) != spike:
            bad.append((r.route, r.week_of.date(), "flag"))
    return len(oracle), bad


class TestPhase1(unittest.TestCase):
    def test_matches_naive_oracle_on_all_route_weeks(self):
        n, bad = compare_to_naive(SHIP, WEEKLY)
        self.assertEqual(n, len(WEEKLY))
        self.assertEqual(bad, [])

    def test_weeks_are_mondays_and_totals_conserved(self):
        self.assertTrue((WEEKLY["week_of"].dt.weekday == 0).all())
        self.assertAlmostEqual(WEEKLY["total_cost"].sum(), SHIP["freight_cost_inr"].sum())
        self.assertAlmostEqual(WEEKLY["total_tonne_km"].sum(),
                               (SHIP["quantity_tonnes"] * SHIP["distance_km"]).sum())

    def test_no_look_ahead(self):
        cutoff = T("2025-01-06")
        mutated = SHIP.copy()
        wk = mutated["shipment_date"] - pd.to_timedelta(mutated["shipment_date"].dt.weekday, unit="D")
        mutated.loc[wk >= cutoff, "freight_cost_inr"] *= 3
        a, b = WEEKLY.set_index(["route", "week_of"]), fa.build_weekly_table(mutated).set_index(["route", "week_of"])
        early = a.index.get_level_values("week_of") <= cutoff
        pd.testing.assert_series_equal(a.loc[early, "historical_avg_8w"], b.loc[early, "historical_avg_8w"])

    def test_peer_average_excludes_the_route_itself(self):
        mutated = SHIP.copy()
        mask = (mutated["origin"] == "Mumbai") & (mutated["destination"] == "Pune")
        mutated.loc[mask, "freight_cost_inr"] *= 2
        a = WEEKLY[WEEKLY["route"] == "Mumbai-Pune"]["peer_avg_this_week"].to_numpy()
        b = fa.build_weekly_table(mutated).query("route == 'Mumbai-Pune'")["peer_avg_this_week"].to_numpy()
        np.testing.assert_allclose(a, b)

    def test_history_uses_only_prior_weeks_and_warmup_is_not_padded(self):
        first = WEEKLY[WEEKLY["n_prior_weeks"] == 0]
        self.assertEqual(len(first), WEEKLY["route"].nunique())
        self.assertTrue(first["historical_avg_8w"].isna().all())
        self.assertEqual(WEEKLY["n_prior_weeks"].max(), fa.HIST_WINDOW)

    def test_calendar_gap_handled_like_naive_oracle(self):
        wk = SHIP["shipment_date"] - pd.to_timedelta(SHIP["shipment_date"].dt.weekday, unit="D")
        gap = ((SHIP["origin"] == "Chennai") & (wk == T("2024-05-06")))
        ship2 = SHIP[~gap]
        n, bad = compare_to_naive(ship2, fa.build_weekly_table(ship2))
        self.assertEqual(bad, [])

    def test_bad_input_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.csv"
            SHIP.assign(freight_cost_inr=-1).to_csv(p, index=False)
            with self.assertRaises(ValueError):
                fa.load_shipments(p)
            SHIP.drop(columns=["distance_km"]).to_csv(p, index=False)
            with self.assertRaises(ValueError):
                fa.load_shipments(p)


# ----------------------------------------------------------------------------
# 3. Verifier matrix (expected values written by hand from the note texts)
# ----------------------------------------------------------------------------
VERIFIER_CASES = [
    # note, route, week_of, expected_ok, check expected to fail (None if ok)
    ("N001", "Chennai-Bangalore", "2025-02-24", True, None),
    ("N001", "Chennai-Bangalore", "2025-03-03", True, None),
    ("N001", "Chennai-Bangalore", "2025-03-10", False, "period_overlaps_week"),   # flood ended Mar 8
    ("N001", "Chennai-Bangalore", "2025-02-17", False, "not_dated_after_week"),   # note dated Feb 24
    ("N001", "Mumbai-Pune",       "2025-02-24", False, "route_applies"),          # wrong route
    ("N002", "Ahmedabad-Mumbai",  "2025-01-20", True, None),
    ("N002", "Ahmedabad-Mumbai",  "2025-01-27", False, "period_overlaps_week"),   # festival week is over
    ("N002", "Chennai-Bangalore", "2025-01-20", False, "route_applies"),
    ("N003", "Mumbai-Delhi",      "2025-05-05", True, None),
    ("N003", "Mumbai-Delhi",      "2025-05-12", True, None),
    ("N003", "Mumbai-Delhi",      "2025-04-28", False, "not_dated_after_week"),   # before the hike
    ("N003", "Mumbai-Delhi",      "2025-06-16", False, "within_lookback"),        # beyond 21-day lookback
    ("N004", "Mumbai-Pune",       "2024-03-18", False, "describes_cost_increase"),
    ("N005", "Mumbai-Delhi",      "2024-07-29", False, "describes_cost_increase"),
    ("N006", "Mumbai-Pune",       "2025-09-22", False, "describes_cost_increase"),
    ("N006", "Mumbai-Pune",       "2025-09-15", False, "not_dated_after_week"),   # sample-file case
    ("N007", "Delhi-Jaipur",      "2024-05-20", False, "describes_cost_increase"),
    ("N008", "Kolkata-Bhubaneswar", "2025-06-09", False, "describes_cost_increase"),
    ("N009", "Chennai-Bangalore", "2025-03-17", False, "describes_cost_increase"),
    ("N010", "Mumbai-Pune",       "2025-10-27", False, "describes_cost_increase"),
]


class TestVerifier(unittest.TestCase):
    def test_matrix(self):
        for nid, route, week, ok, failing in VERIFIER_CASES:
            with self.subTest(note=nid, route=route, week=week):
                v = fa.verify_note(NOTE.loc[nid], route, T(week))
                self.assertEqual(v["ok"], ok, v)
                if failing:
                    self.assertIn(failing, v["failed"])

    def test_window_parsing(self):
        s, e, basis = fa.note_window(T("2025-02-24"), NOTE.loc["N001", "note"])
        self.assertEqual((s, e, basis), (T("2025-02-24"), T("2025-03-08"), "explicit-range"))
        self.assertEqual(fa.note_window(T("2025-05-05"), NOTE.loc["N003", "note"])[1:], (None, "open-ended"))
        self.assertEqual(fa.note_window(T("2025-01-20"), NOTE.loc["N002", "note"])[2], "note-week")


# ----------------------------------------------------------------------------
# 4. Judge guard tests (LLM stubbed)
# ----------------------------------------------------------------------------
class StubJudge(fa.LLMJudge):
    """LLMJudge whose network call is replaced by scripted outcomes."""

    def __init__(self, script, cache_path, **kw):
        super().__init__(cache_path, sleep=lambda s: None, rate_limit_sleep=0, **kw)
        self.script, self.calls = list(script), []

    def _generate(self, model, prompt):
        self.calls.append(model)
        out = self.script.pop(0)
        if isinstance(out, Exception):
            raise out
        text = out if isinstance(out, str) else json.dumps(out)
        return SimpleNamespace(text=text, usage_metadata=SimpleNamespace(
            prompt_token_count=100, candidates_token_count=20))


def spike_row(route, week):
    r = WEEKLY[(WEEKLY["route"] == route) & (WEEKLY["week_of"] == T(week))].iloc[0].copy()
    return r


def verdict(flag, note_id="", reason="stub"):
    return {"flagged": flag, "matched_note_id": note_id, "reason": reason}


class TestJudgeGuards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name) / "c.json"

    def tearDown(self):
        self.tmp.cleanup()

    def run_case(self, route, week, llm_verdict, cands=None):
        spike = spike_row(route, week)
        cands = RETRIEVER.candidates(route, spike["week_of"]) if cands is None else cands
        judge = StubJudge([llm_verdict], self.cache)
        return fa.decide(spike, cands, judge.judge(spike, cands))

    def test_cannot_cite_unretrieved_note(self):
        r = self.run_case("Chennai-Bangalore", "2025-03-10", verdict(fa.FLAG_JUSTIFIED, "N003"))
        self.assertEqual((r["flagged"], r["matched_note_id"]), (fa.FLAG_UNEXPLAINED, ""))
        self.assertIn("not among the retrieved", r["reason"])

    def test_cannot_use_expired_note(self):
        r = self.run_case("Chennai-Bangalore", "2025-03-10", verdict(fa.FLAG_JUSTIFIED, "N001"))
        self.assertEqual(r["flagged"], fa.FLAG_UNEXPLAINED)
        self.assertIn("period_overlaps_week", r["reason"])

    def test_cannot_use_wrong_route_note(self):
        forced = NOTES[NOTES["note_id"] == "N001"].assign(similarity=1.0)
        r = self.run_case("Mumbai-Pune", "2025-02-24", verdict(fa.FLAG_JUSTIFIED, "N001"), cands=forced)
        self.assertEqual(r["flagged"], fa.FLAG_UNEXPLAINED)
        self.assertIn("route_applies", r["reason"])

    def test_cannot_use_future_dated_note(self):
        forced = NOTES[NOTES["note_id"] == "N001"].assign(similarity=1.0)
        r = self.run_case("Chennai-Bangalore", "2025-02-17", verdict(fa.FLAG_JUSTIFIED, "N001"), cands=forced)
        self.assertEqual(r["flagged"], fa.FLAG_UNEXPLAINED)
        self.assertIn("not_dated_after_week", r["reason"])

    def test_cannot_justify_with_no_increase_note(self):
        forced = NOTES[NOTES["note_id"] == "N010"].assign(similarity=1.0)
        r = self.run_case("Mumbai-Pune", "2025-10-27", verdict(fa.FLAG_JUSTIFIED, "N010"), cands=forced)
        self.assertEqual(r["flagged"], fa.FLAG_UNEXPLAINED)
        self.assertIn("describes_cost_increase", r["reason"])

    def test_label_drift_is_treated_as_unexplained(self):
        r = self.run_case("Chennai-Bangalore", "2025-03-03", verdict("justified", "N001"))
        self.assertEqual(r["flagged"], fa.FLAG_UNEXPLAINED)

    def test_legitimate_justification_is_grounded_in_note_text(self):
        r = self.run_case("Chennai-Bangalore", "2025-03-03", verdict(fa.FLAG_JUSTIFIED, "N001", "free prose"))
        self.assertEqual((r["flagged"], r["matched_note_id"]), (fa.FLAG_JUSTIFIED, "N001"))
        self.assertIn("Feb 24 to Mar 8", r["reason"])      # note text, not LLM prose
        self.assertNotIn("free prose", r["reason"])
        self.assertIn("2025-02-24", r["reason"])

    def test_all_routes_note_on_peer_only_flag_gets_caveat(self):
        r = self.run_case("Chennai-Bangalore", "2025-05-05", verdict(fa.FLAG_JUSTIFIED, "N003"))
        self.assertEqual(r["flagged"], fa.FLAG_JUSTIFIED)
        self.assertIn("Caveat", r["reason"])

    def test_no_candidates_never_calls_llm_and_is_unexplained(self):
        spike = spike_row("Delhi-Jaipur", "2024-11-11")
        cands = RETRIEVER.candidates(spike["route"], spike["week_of"])
        self.assertTrue(cands.empty)
        r = fa.decide(spike, cands, None)
        self.assertEqual((r["flagged"], r["source"]), (fa.FLAG_UNEXPLAINED, "rule"))

    def test_malformed_json_retries_then_falls_back_to_next_model(self):
        spike = spike_row("Chennai-Bangalore", "2025-03-03")
        cands = RETRIEVER.candidates(spike["route"], spike["week_of"])
        j = StubJudge(["not json"] * 3 + [verdict(fa.FLAG_JUSTIFIED, "N001")], self.cache)
        res = j.judge(spike, cands)
        self.assertEqual(res["model"], fa.MODEL_CHAIN[1])
        self.assertEqual(j.stats["failed_attempts"], 3)

    def test_404_skips_model_without_retrying(self):
        spike = spike_row("Chennai-Bangalore", "2025-03-03")
        cands = RETRIEVER.candidates(spike["route"], spike["week_of"])
        j = StubJudge([RuntimeError("404 NOT_FOUND"), verdict(fa.FLAG_JUSTIFIED, "N001")], self.cache)
        res = j.judge(spike, cands)
        self.assertEqual(j.calls, fa.MODEL_CHAIN[:2])
        self.assertEqual(res["model"], fa.MODEL_CHAIN[1])

    def test_pinned_model_never_falls_back_and_retries_through_503(self):
        spike = spike_row("Chennai-Bangalore", "2025-03-03")
        cands = RETRIEVER.candidates(spike["route"], spike["week_of"])
        pin = fa.MODEL_CHAIN[1]
        j = StubJudge([RuntimeError("503 UNAVAILABLE")] * 4 + [verdict(fa.FLAG_JUSTIFIED, "N001")], self.cache,
                      model_chain=[pin], max_retries=8)
        res = j.judge(spike, cands)
        self.assertEqual(set(j.calls), {pin})            # never touched another model
        self.assertEqual((res["model"], j.stats["failed_attempts"]), (pin, 4))

    def test_pinned_model_gives_up_instead_of_switching(self):
        spike = spike_row("Chennai-Bangalore", "2025-03-03")
        cands = RETRIEVER.candidates(spike["route"], spike["week_of"])
        j = StubJudge([RuntimeError("503 UNAVAILABLE")] * 3, self.cache,
                      model_chain=[fa.MODEL_CHAIN[1]], max_retries=3)
        self.assertIsNone(j.judge(spike, cands))
        self.assertEqual(set(j.calls), {fa.MODEL_CHAIN[1]})

    def test_total_failure_is_error_not_a_guess_unless_fallback_requested(self):
        spike = spike_row("Chennai-Bangalore", "2025-03-03")
        cands = RETRIEVER.candidates(spike["route"], spike["week_of"])
        self.assertEqual(fa.decide(spike, cands, None)["status"], "error")
        fb = fa.decide(spike, cands, None, fallback="verifier")
        self.assertEqual((fb["model"], fb["flagged"]), ("verifier-fallback", fa.FLAG_JUSTIFIED))

    def test_cache_prevents_repeat_calls_and_token_accounting(self):
        spike = spike_row("Chennai-Bangalore", "2025-03-03")
        cands = RETRIEVER.candidates(spike["route"], spike["week_of"])
        j = StubJudge([verdict(fa.FLAG_JUSTIFIED, "N001")], self.cache)
        j.judge(spike, cands)
        self.assertEqual((j.stats["input_tokens"], j.stats["output_tokens"]), (100, 20))
        j2 = StubJudge([], self.cache)           # empty script: any live call would raise
        again = j2.judge(spike, cands)
        self.assertEqual((j2.stats["cache_hits"], len(j2.calls)), (1, 0))
        self.assertTrue(again["source"].startswith("cache"))

    def test_cache_key_is_order_independent_and_text_sensitive(self):
        spike = spike_row("Chennai-Bangalore", "2025-03-17")
        c = RETRIEVER.candidates(spike["route"], spike["week_of"])
        k1 = fa.LLMJudge.cache_key(spike, c)
        self.assertEqual(k1, fa.LLMJudge.cache_key(spike, c.iloc[::-1]))
        edited = c.copy()
        edited.iloc[0, edited.columns.get_loc("note")] = "changed text"
        self.assertNotEqual(k1, fa.LLMJudge.cache_key(spike, edited))


# ----------------------------------------------------------------------------
# 5. Contract mutation tests
# ----------------------------------------------------------------------------
SUB_PATH = fa.OUT_DIR / "final_submission.csv"
SAMPLE = fa.DATA_DIR / "sample_output_format_v2.csv"


class TestContract(unittest.TestCase):
    def mutate(self, fn):
        df = pd.read_csv(SUB_PATH, dtype=str, keep_default_na=False)
        df = fn(df)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.csv"
            df.to_csv(p, index=False)
            return fa.validate_submission(p, SAMPLE, NOTES, WEEKLY)

    def test_real_output_is_clean(self):
        self.assertEqual(fa.validate_submission(SUB_PATH, SAMPLE, NOTES, WEEKLY), [])

    def test_header_matches_sample_exactly(self):
        sample_cols = SAMPLE.read_text().splitlines()[0].split(",")
        self.assertEqual(list(pd.read_csv(SUB_PATH, nrows=0).columns), sample_cols)

    def test_catches_each_corruption(self):
        def set_first_justified(col, val):
            def f(df):
                i = df.index[df["flagged"] == fa.FLAG_JUSTIFIED][0]
                df.loc[i, col] = val
                return df
            return f
        self.assertTrue(self.mutate(set_first_justified("matched_note_id", "N010")))   # wrong note
        self.assertTrue(self.mutate(set_first_justified("matched_note_id", "")))       # blank cite
        self.assertTrue(self.mutate(set_first_justified("matched_note_id", "N999")))   # unknown note
        self.assertTrue(self.mutate(set_first_justified("flagged", "ERROR")))
        self.assertTrue(self.mutate(lambda d: d.drop(columns=["reason"])))
        self.assertTrue(self.mutate(lambda d: d.iloc[1:]))                             # missing row

        def unexplained_with_note(df):
            i = df.index[df["flagged"] == fa.FLAG_UNEXPLAINED][0]
            df.loc[i, "matched_note_id"] = "N001"
            return df
        self.assertTrue(self.mutate(unexplained_with_note))


# ----------------------------------------------------------------------------
# 6 + 7. Gold-set scoring and diagnostics
# ----------------------------------------------------------------------------
def gold_scoring() -> dict:
    gold = pd.read_csv(fa.ROOT / "eval" / "gold_labels.csv", keep_default_na=False)
    sub = pd.read_csv(SUB_PATH, keep_default_na=False)
    sub_idx = sub.set_index(["route", "week_of"])
    # Every spike not hand-labelled has no note naming its route (or All Routes) in the
    # 21-day window; gold for those is "Yes" by construction (auditable from context_notes.csv).
    labelled = set(zip(gold["route"], gold["week_of"]))
    rest = [(r, w) for (r, w) in sub_idx.index if (r, w) not in labelled]
    rest_ok = sum(sub_idx.loc[k, "flagged"] == fa.FLAG_UNEXPLAINED and sub_idx.loc[k, "matched_note_id"] == ""
                  for k in rest)

    def score(lenient: bool):
        tp = fp = fn = tn = cite_ok = 0
        for g in gold.itertuples():
            exp_just = g.gold_flagged == fa.FLAG_JUSTIFIED and (lenient or g.certainty == "clear")
            got = sub_idx.loc[(g.route, g.week_of)]
            got_just = got["flagged"] == fa.FLAG_JUSTIFIED
            tp += exp_just and got_just
            fp += (not exp_just) and got_just
            fn += exp_just and not got_just
            tn += (not exp_just) and not got_just
            cite_ok += (not got_just) or got["matched_note_id"] == g.gold_note_id
        n = len(gold)
        return {"rows": n, "accuracy": round((tp + tn) / n, 3), "justified_precision": round(tp / max(tp + fp, 1), 3),
                "justified_recall": round(tp / max(tp + fn, 1), 3), "tp": int(tp), "fp": int(fp), "fn": int(fn),
                "tn": int(tn), "justified_rows_citing_correct_note": f"{int(cite_ok)}/{n}"}
    return {"lenient (ambiguous N003 weeks = justified)": score(True),
            "strict (ambiguous N003 weeks = unexplained)": score(False),
            "unlabelled_spikes_correctly_unexplained": f"{rest_ok}/{len(rest)}"}


def retrieval_diagnostics() -> dict:
    spikes = WEEKLY[WEEKLY["is_spike"]]
    hard_fail = {"route_applies", "not_dated_after_week", "within_lookback"}
    invalid_in_top3 = missed_gold = 0
    gold = pd.read_csv(fa.ROOT / "eval" / "gold_labels.csv", keep_default_na=False)
    gold_j = {(g.route, g.week_of): g.gold_note_id for g in gold.itertuples() if g.gold_flagged == fa.FLAG_JUSTIFIED}
    for s in spikes.itertuples():
        q = RETRIEVER._embed([RETRIEVER.query_for(s.route, s.week_of)])[0]
        m = RETRIEVER.note_vecs
        sims = m @ q / (np.linalg.norm(m, axis=1) * np.linalg.norm(q) + 1e-12)
        top = RETRIEVER.notes.iloc[np.argsort(-sims, kind="stable")[:fa.TOP_K]]
        fails = [hard_fail & set(fa.verify_note(n, s.route, s.week_of)["failed"]) for _, n in top.iterrows()]
        invalid_in_top3 += any(fails)
        key = (s.route, f"{s.week_of:%Y-%m-%d}")
        if key in gold_j and gold_j[key] not in set(top["note_id"]):
            missed_gold += 1
    cand_sizes = [len(RETRIEVER.candidates(s.route, s.week_of)) for s in spikes.itertuples()]
    return {"backend": RETRIEVER.backend,
            "spikes": len(spikes),
            "unfiltered_top3_contains_route_or_date_invalid_note": f"{invalid_in_top3}/{len(spikes)}",
            "unfiltered_top3_misses_gold_note": f"{missed_gold}/{len(gold_j)}",
            "candidates_per_spike_max": max(cand_sizes),
            "spikes_with_zero_candidates": int(sum(c == 0 for c in cand_sizes)),
            "note": "top-k never binds (max candidates < TOP_K): the hard filter does the real work"}


def threshold_sensitivity() -> list[dict]:
    rows, w = [], WEEKLY
    for t in (1.05, 1.10, 1.15, 1.20):
        bh, bp = w["cost_per_tonne_km"] > w["historical_avg_8w"] * t, w["cost_per_tonne_km"] > w["peer_avg_this_week"] * t
        rows.append({"threshold": t, "flagged": int((bh | bp).sum()), "history_only_or_both": int(bh.sum()),
                     "peers_only": int((bp & ~bh).sum()),
                     "mumbai_pune": int(((bh | bp) & (w["route"] == "Mumbai-Pune")).sum())})
    return rows


def verifier_vs_llm_agreement() -> dict:
    d = pd.read_csv(fa.OUT_DIR / "audit_detail.csv", keep_default_na=False)
    d = d[d["status"] == "ok"]
    ver_just = d["verifier_pass_ids"] != ""
    llm_just = d["llm_flagged"] == fa.FLAG_JUSTIFIED
    same_note = d.apply(lambda r: r["llm_flagged"] != fa.FLAG_JUSTIFIED or r["llm_note_id"] in r["verifier_pass_ids"].split(","), axis=1)
    return {"rows_judged_by_llm": len(d), "verifier_and_llm_agree": int((ver_just == llm_just).sum()),
            "llm_justified_but_verifier_rejects": int((llm_just & ~ver_just).sum()),
            "verifier_accepts_but_llm_rejects": int((~llm_just & ver_just).sum()),
            "llm_cited_note_passes_verifier": bool(same_note.all())}


def token_estimate() -> dict:
    """ESTIMATE only (chars/4). The measured numbers come from outputs/run_log.json on a live run."""
    in_chars = out_chars = calls = 0
    for s in WEEKLY[WEEKLY["is_spike"]].itertuples():
        c = RETRIEVER.candidates(s.route, s.week_of)
        if c.empty:
            continue
        calls += 1
        in_chars += len(fa.build_prompt(pd.Series(s._asdict()), c))
        out_chars += 230  # typical JSON verdict in the original run (~1-2 sentences)
    i, o = in_chars // 4, out_chars // 4
    return {"basis": "ESTIMATE (chars/4), not measured", "llm_calls_needed_per_clean_run": calls,
            "input_tokens_est": i, "output_tokens_est": o,
            "cost_usd_at_published_paid_rate": round(i / 1e6 * 0.25 + o / 1e6 * 1.50, 5),
            "rate_used": "$0.25 in / $1.50 out per 1M tokens (Gemini 3.1 Flash-Lite Preview, ai.google.dev/pricing); free tier = $0"}


def main() -> int:
    suite = unittest.TestSuite()
    for case in (TestPhase1, TestVerifier, TestJudgeGuards, TestContract):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(case))
    res = unittest.TextTestRunner(verbosity=1).run(suite)
    report = {"unit_tests": {"run": res.testsRun, "failures": len(res.failures), "errors": len(res.errors)},
              "gold_scoring": gold_scoring(), "retrieval": retrieval_diagnostics(),
              "threshold_sensitivity": threshold_sensitivity(),
              "verifier_vs_llm_on_original_verdicts": verifier_vs_llm_agreement(),
              "token_estimate": token_estimate()}
    (fa.OUT_DIR / "eval_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if res.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
