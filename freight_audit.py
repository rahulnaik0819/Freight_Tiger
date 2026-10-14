"""FreightTiger freight-cost anomaly auditor.

Pipeline (same stages as the original notebook, now one runnable, testable module):

  1. Phase 1 (deterministic): weekly cost per tonne-km per route, trailing 8-week
     own-history baseline (strictly prior weeks) and same-route_type peer baseline
     (self excluded); flag a week when it exceeds either baseline by THRESHOLD.
  2. Retrieval: hard filter on route + date window, then rank by similarity.
  3. LLM judge (temperature 0, JSON schema) decides whether a retrieved note
     justifies the rise.
  4. Deterministic verifier: a justification only stands if the cited note applies
     to the route, is not dated after the spike week, its stated period overlaps
     the spike week, and its text describes a cost increase. The LLM can never
     create a justification the verifier rejects.
  5. Reasons for the CSV are assembled from verified note fields, not free LLM prose.

Usage:
    python freight_audit.py run                # full run (uses LLM cache if present)
    python freight_audit.py run --no-cache     # force live LLM calls
    python freight_audit.py validate           # check the CSV contract only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
THRESHOLD = 1.10          # flag when cost > baseline * THRESHOLD (unchanged from original)
HIST_WINDOW = 8           # trailing weeks for own-history baseline (brief)
WINDOW_DAYS = 21          # notes may pre-date the spike week by up to this many days
TOP_K = 3                 # max notes shown to the judge
PROMPT_VERSION = "v1"     # bump when the prompt/schema changes (invalidates cache)
MODEL_CHAIN = [           # unchanged from original
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3-flash-preview",
]
MAX_RETRIES = 3
COOLDOWN_SECONDS = 300

CONTRACT_COLUMNS = [
    "route", "week_of", "cost_per_tonne_km", "vs_own_history",
    "vs_similar_routes", "flagged", "matched_note_id", "reason",
]
FLAG_UNEXPLAINED = "Yes"
FLAG_JUSTIFIED = "No (justified)"
NO_NOTE_REASON = ("No matching note found for this route or date range. "
                  "Cost rise looks unexplained and worth a human review.")


# ----------------------------------------------------------------------------
# Phase 1: deterministic engine
# ----------------------------------------------------------------------------
def load_shipments(path: Path) -> pd.DataFrame:
    """Read and validate shipment_records.csv."""
    required = ["origin", "destination", "route_type", "quantity_tonnes",
                "distance_km", "freight_cost_inr", "shipment_date"]
    df = pd.read_csv(path)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    df["shipment_date"] = pd.to_datetime(df["shipment_date"], errors="coerce")
    if df[required].isna().any().any():
        bad = df[required].isna().sum()
        raise ValueError(f"{path.name}: null/unparseable values:\n{bad[bad > 0]}")
    for col in ["quantity_tonnes", "distance_km", "freight_cost_inr"]:
        if (df[col] <= 0).any():
            raise ValueError(f"{path.name}: non-positive values in {col}")
    return df


def build_weekly_table(shipments: pd.DataFrame) -> pd.DataFrame:
    """Weekly (Mon-Sun) cost per tonne-km per route with both baselines and flags."""
    df = shipments.copy()
    df["route"] = df["origin"].str.strip() + "-" + df["destination"].str.strip()
    if (df.groupby("route")["route_type"].nunique() > 1).any():
        raise ValueError("A route maps to more than one route_type")
    df["week_of"] = (df["shipment_date"]
                     - pd.to_timedelta(df["shipment_date"].dt.weekday, unit="D")).dt.normalize()
    df["tonne_km"] = df["quantity_tonnes"] * df["distance_km"]

    w = (df.groupby(["route", "route_type", "week_of"], as_index=False)
           .agg(total_cost=("freight_cost_inr", "sum"),
                total_tonne_km=("tonne_km", "sum")))
    w["cost_per_tonne_km"] = w["total_cost"] / w["total_tonne_km"]
    w = w.sort_values(["route", "week_of"], kind="mergesort").reset_index(drop=True)

    # Own history: mean of the (up to) 8 calendar weeks strictly before this week.
    # Reindexing to a full weekly calendar keeps "8 weeks" meaning calendar weeks even
    # if a route has a gap; missing weeks are skipped, never padded.
    hist = pd.Series(np.nan, index=w.index)
    n_prior = pd.Series(0, index=w.index)
    for _, g in w.groupby("route", sort=False):
        s = g.set_index("week_of")["cost_per_tonne_km"]
        full = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="7D")).shift(1)
        roll = full.rolling(HIST_WINDOW, min_periods=1)
        hist.loc[g.index] = roll.mean().reindex(s.index).to_numpy()
        n_prior.loc[g.index] = roll.count().reindex(s.index).fillna(0).to_numpy()
    w["historical_avg_8w"] = hist
    w["n_prior_weeks"] = n_prior.astype(int)

    # Peers: same route_type, same week, every other route.
    grp = w.groupby(["route_type", "week_of"])["cost_per_tonne_km"]
    peer_sum = grp.transform("sum") - w["cost_per_tonne_km"]
    peer_n = grp.transform("count") - 1
    w["peer_avg_this_week"] = np.where(peer_n > 0, peer_sum / peer_n.where(peer_n > 0), np.nan)

    cpt, h, p = w["cost_per_tonne_km"], w["historical_avg_8w"], w["peer_avg_this_week"]
    w["overrun_vs_history_pct"] = (cpt / h - 1) * 100
    w["overrun_vs_peers_pct"] = (cpt / p - 1) * 100
    by_hist, by_peer = cpt > h * THRESHOLD, cpt > p * THRESHOLD
    w["is_spike"] = by_hist | by_peer
    w["flag_basis"] = np.select([by_hist & by_peer, by_hist, by_peer],
                                ["history+peers", "history", "peers"], default="")
    return w


# ----------------------------------------------------------------------------
# Notes, retrieval
# ----------------------------------------------------------------------------
def load_notes(path: Path) -> pd.DataFrame:
    notes = pd.read_csv(path)
    for c in ["note_id", "date", "applies_to", "note"]:
        if c not in notes.columns:
            raise ValueError(f"{path.name}: missing column {c}")
    notes["date"] = pd.to_datetime(notes["date"], errors="raise")
    if notes["note_id"].duplicated().any():
        raise ValueError("Duplicate note_id in context notes")
    notes["applies_to"] = notes["applies_to"].str.strip()
    notes["search_text"] = ("Route: " + notes["applies_to"] + " | Date: "
                            + notes["date"].dt.strftime("%Y-%m-%d") + " | Note: " + notes["note"])
    return notes


class Retriever:
    """Hard filter (route + date window) -> similarity rank -> top-k.

    backend: 'minilm' (sentence-transformers all-MiniLM-L6-v2, as in the original),
             'tfidf'  (deterministic, offline), or 'auto' (minilm, else tfidf).
    """

    def __init__(self, notes: pd.DataFrame, backend: str = "auto"):
        self.notes = notes.reset_index(drop=True)
        self.backend, self._embed = self._init_backend(backend)
        self.note_vecs = self._embed(self.notes["search_text"].tolist())

    def _init_backend(self, backend: str):
        if backend in ("auto", "minilm"):
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer("all-MiniLM-L6-v2")
                return "minilm", lambda texts: np.asarray(model.encode(texts))
            except Exception as exc:  # no package / no network
                if backend == "minilm":
                    raise
                print(f"[retriever] MiniLM unavailable ({type(exc).__name__}); using TF-IDF.")
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(self.notes["search_text"])
        return "tfidf", lambda texts: vec.transform(texts).toarray()

    @staticmethod
    def query_for(route: str, week_of: pd.Timestamp) -> str:
        return (f"Freight cost per tonne-km rose on route {route} in the week of "
                f"{week_of:%Y-%m-%d}. Possible reasons: fuel price, toll, weather "
                f"disruption, festival surcharge.")

    def candidates(self, route: str, week_of: pd.Timestamp) -> pd.DataFrame:
        n = self.notes
        week_end = week_of + pd.Timedelta(days=6)
        mask = (n["applies_to"].str.lower().isin([route.lower(), "all routes"])
                & (n["date"] >= week_of - pd.Timedelta(days=WINDOW_DAYS))
                & (n["date"] <= week_end))
        idx = n.index[mask].tolist()
        if not idx:
            return n.iloc[0:0].assign(similarity=[])
        q = self._embed([self.query_for(route, week_of)])[0]
        mat = self.note_vecs[idx]
        sims = mat @ q / (np.linalg.norm(mat, axis=1) * np.linalg.norm(q) + 1e-12)
        order = sorted(range(len(idx)), key=lambda i: (-round(float(sims[i]), 6), n.loc[idx[i], "note_id"]))
        top = [idx[i] for i in order[:TOP_K]]
        return n.loc[top].assign(similarity=[float(sims[order[k]]) for k in range(len(top))])


# ----------------------------------------------------------------------------
# Deterministic verifier (independent of the LLM)
# ----------------------------------------------------------------------------
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_TEXT_RANGE = re.compile(r"([A-Za-z]{3,9})\.?\s+(\d{1,2})\s*(?:to|through|until|-|–)\s*"
                         r"(?:([A-Za-z]{3,9})\.?\s+)?(\d{1,2})\b")
_ISO_RANGE = re.compile(r"(\d{4}-\d{2}-\d{2})\s*(?:to|through|until|-|–)\s*(\d{4}-\d{2}-\d{2})")
_OPEN_ENDED = re.compile(r"\b(starting|starts|beginning|from this week|effective|onwards?|going forward)\b", re.I)
_NO_INCREASE = re.compile(
    r"not (?:significantly )?affected|without a rate change|no rate change|"
    r"not part of (?:this|the) dataset|remained (?:stable|normal)|"
    r"no (?:major|significant) disruptions?|returned to normal|improved road conditions|"
    r"costs? (?:were|was|remained) (?:not|unchanged)", re.I)
_INCREASE = re.compile(
    r"\b(higher|rise[sn]?|rose|rising|increase[sd]?|surcharge|pushing up|hike[sd]?|"
    r"detours?|costlier|pricier)\b", re.I)
_PCT_RANGE = re.compile(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*%")


def note_window(note_date: pd.Timestamp, text: str):
    """Period a note describes -> (start, end|None for open-ended, basis)."""
    m = _ISO_RANGE.search(text)
    if m:
        return pd.Timestamp(m.group(1)), pd.Timestamp(m.group(2)), "explicit-range"
    for m in _TEXT_RANGE.finditer(text):
        m1 = _MONTHS.get(m.group(1)[:3].lower())
        m2 = _MONTHS.get((m.group(3) or m.group(1))[:3].lower())
        if m1 and m2:
            start = pd.Timestamp(year=note_date.year, month=m1, day=int(m.group(2)))
            end = pd.Timestamp(year=note_date.year, month=m2, day=int(m.group(4)))
            if end < start:
                end = end + pd.DateOffset(years=1)
            return start, end, "explicit-range"
    if _OPEN_ENDED.search(text):
        return note_date, None, "open-ended"
    monday = note_date - pd.Timedelta(days=note_date.weekday())
    return monday, monday + pd.Timedelta(days=6), "note-week"


def supports_increase(text: str) -> bool:
    """True only if the note describes a cost rise and does not say costs were unaffected."""
    return bool(_INCREASE.search(text)) and not _NO_INCREASE.search(text)


def verify_note(note: pd.Series, route: str, week_of: pd.Timestamp) -> dict:
    """Independent checks that a note legitimately justifies a spike for route/week."""
    week_end = week_of + pd.Timedelta(days=6)
    start, end, basis = note_window(note["date"], note["note"])
    checks = {
        "route_applies": note["applies_to"].lower() in (route.lower(), "all routes"),
        "not_dated_after_week": bool(note["date"] <= week_end),
        "within_lookback": bool(note["date"] >= week_of - pd.Timedelta(days=WINDOW_DAYS)),
        "period_overlaps_week": bool(start <= week_end and (end is None or end >= week_of)),
        "describes_cost_increase": supports_increase(note["note"]),
    }
    return {"ok": all(checks.values()), "checks": checks,
            "failed": [k for k, v in checks.items() if not v],
            "window": f"{start:%Y-%m-%d}..{'open' if end is None else format(end, '%Y-%m-%d')} ({basis})"}


# ----------------------------------------------------------------------------
# LLM judge
# ----------------------------------------------------------------------------
def build_prompt(spike: pd.Series, cands: pd.DataFrame) -> str:
    context = "".join(
        f"- ID: {n.note_id} | Date: {n.date:%Y-%m-%d} | Route: {n.applies_to} | Text: {n.note}\n"
        for n in cands.itertuples())
    return textwrap.dedent(f"""
        You are an AI auditor for a freight company. A route's cost has spiked.
        Determine if the spike is justified based ONLY on the context notes provided.

        Spike Details:
        - Route: {spike['route']}
        - Week Of (Monday): {spike['week_of']:%Y-%m-%d}
        - Current Cost per tonne-km: {spike['cost_per_tonne_km']:.2f}
        - Historical Avg: {spike['historical_avg_8w']:.2f}
        - Peer Avg: {spike['peer_avg_this_week']:.2f}

        Retrieved Context Notes:
        {context}

        Rules:
        1. The note MUST apply to the route (or 'All Routes').
        2. The period described IN THE NOTE TEXT must cover the spike week.
        3. The note MUST explain a cost INCREASE (not "returned to normal", "not affected", "no rate change", "stable").
        4. If no note perfectly matches, it is NOT justified (flagged: Yes). DO NOT HALLUCINATE.
        """)


def get_api_key() -> str | None:
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    try:  # Colab secrets
        from google.colab import userdata  # type: ignore
        return userdata.get("GEMINI_API_KEY")
    except Exception:
        return None


def parse_llm_json(text: str) -> dict:
    """Parse the judge's JSON; tolerate code fences; reject anything malformed."""
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip())
    obj = json.loads(cleaned)
    if not isinstance(obj, dict) or not {"flagged", "matched_note_id", "reason"} <= obj.keys():
        raise ValueError("LLM JSON missing required keys")
    return obj


class LLMJudge:
    """Gemini judge with retries, model fallback, cooldowns, token accounting and a verdict cache.

    The cache doubles as the checkpoint: it is written after every live call, so an
    interrupted run resumes without repeating (or paying for) finished rows.
    """

    def __init__(self, cache_path: Path, use_cache: bool = True, client=None,
                 model_chain=MODEL_CHAIN, sleep=time.sleep, rate_limit_sleep: float = 1.0,
                 max_retries: int = MAX_RETRIES):
        self.cache_path, self.use_cache = Path(cache_path), use_cache
        self.max_retries = max_retries
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        self._client, self.model_chain, self._sleep = client, model_chain, sleep
        self.rate_limit_sleep = rate_limit_sleep
        self.cooldowns: dict[str, float] = {}
        self.stats = {"live_calls": 0, "failed_attempts": 0, "cache_hits": 0,
                      "input_tokens": 0, "output_tokens": 0, "models": {}}

    @staticmethod
    def cache_key(spike: pd.Series, cands: pd.DataFrame) -> str:
        payload = {"v": PROMPT_VERSION, "route": spike["route"],
                   "week": f"{spike['week_of']:%Y-%m-%d}",
                   "notes": sorted((n.note_id, hashlib.sha256(n.note.encode()).hexdigest()[:12])
                                   for n in cands.itertuples())}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]

    def _save(self):
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.cache, indent=1, sort_keys=True))
        tmp.replace(self.cache_path)

    def _ensure_client(self):
        if self._client is None:
            key = get_api_key()
            if not key:
                raise RuntimeError("GEMINI_API_KEY not set (env var or Colab secret)")
            from google import genai
            self._client = genai.Client(api_key=key)
        return self._client

    def _generate(self, model: str, prompt: str):
        from google.genai import types
        schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "flagged": types.Schema(type=types.Type.STRING, enum=[FLAG_UNEXPLAINED, FLAG_JUSTIFIED]),
                "matched_note_id": types.Schema(type=types.Type.STRING),
                "reason": types.Schema(type=types.Type.STRING)},
            required=["flagged", "matched_note_id", "reason"])
        return self._ensure_client().models.generate_content(
            model=model, contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0, response_mime_type="application/json", response_schema=schema))

    def judge(self, spike: pd.Series, cands: pd.DataFrame) -> dict | None:
        """Return {'verdict': {...}, 'model', 'source', 'usage'} or None if every model failed."""
        key = self.cache_key(spike, cands)
        if self.use_cache and key in self.cache:
            self.stats["cache_hits"] += 1
            return {**self.cache[key], "source": "cache:" + self.cache[key].get("source", "live")}
        prompt = build_prompt(spike, cands)
        for model in self.model_chain:
            if self.cooldowns.get(model, 0) > time.time():
                continue
            self.cooldowns.pop(model, None)
            for attempt in range(self.max_retries):
                try:
                    resp = self._generate(model, prompt)
                    verdict = parse_llm_json(resp.text)
                    meta = getattr(resp, "usage_metadata", None)
                    # thinking tokens are billed as output, so they are counted with it
                    usage = {"input_tokens": getattr(meta, "prompt_token_count", None) or 0,
                             "output_tokens": (getattr(meta, "candidates_token_count", None) or 0)
                                              + (getattr(meta, "thoughts_token_count", None) or 0)}
                    self.stats["live_calls"] += 1
                    self.stats["input_tokens"] += usage["input_tokens"]
                    self.stats["output_tokens"] += usage["output_tokens"]
                    self.stats["models"][model] = self.stats["models"].get(model, 0) + 1
                    entry = {"verdict": verdict, "model": model, "usage": usage, "source": "live"}
                    self.cache[key] = entry
                    self._save()
                    self._sleep(self.rate_limit_sleep)
                    return entry
                except Exception as exc:
                    self.stats["failed_attempts"] += 1
                    print(f"    [!] {model} attempt {attempt + 1}: {type(exc).__name__} - {str(exc)[:80]}")
                    if "404" in str(exc):
                        break  # retired model -> next in chain
                    self._sleep(min(2 ** attempt, 30))
            else:
                print(f"    [!] {model} exhausted retries; cooling down {COOLDOWN_SECONDS}s")
                self.cooldowns[model] = time.time() + COOLDOWN_SECONDS
        return None


# ----------------------------------------------------------------------------
# Verdict assembly: LLM verdict + deterministic verifier -> final row
# ----------------------------------------------------------------------------
def _fmt_note(note: pd.Series) -> str:
    return f"{note['note_id']}, {note['date']:%Y-%m-%d}"


def decide(spike: pd.Series, cands: pd.DataFrame, llm: dict | None, fallback: str = "abort") -> dict:
    """Combine LLM verdict with the verifier. Justified requires BOTH to agree."""
    week = spike["week_of"]
    row = {"llm_flagged": "", "llm_note_id": "", "llm_reason": "", "model": "", "source": "",
           "verifier_pass_ids": "", "overridden": False, "status": "ok",
           "input_tokens": 0, "output_tokens": 0}

    if cands.empty:
        return {**row, "flagged": FLAG_UNEXPLAINED, "matched_note_id": "", "reason": NO_NOTE_REASON,
                "model": "rule", "source": "rule", "status": "rule"}

    ver = {n["note_id"]: verify_note(n, spike["route"], week) for _, n in cands.iterrows()}
    passing = [nid for nid, v in ver.items() if v["ok"]]
    row["verifier_pass_ids"] = ",".join(passing)
    top = cands.iloc[0]

    if llm is None:
        if fallback == "verifier":  # explicit opt-in: deterministic verdict, clearly labelled
            llm_flag, note_id, llm_reason = (FLAG_JUSTIFIED, passing[0], "") if passing else (FLAG_UNEXPLAINED, "", "")
            row.update(model="verifier-fallback", source="fallback")
        else:
            return {**row, "flagged": "ERROR", "matched_note_id": "", "status": "error",
                    "reason": "LLM unavailable for every model in the chain.", "model": "none"}
    else:
        v = llm["verdict"]
        llm_flag = v.get("flagged", FLAG_UNEXPLAINED)
        llm_flag = llm_flag if llm_flag in (FLAG_UNEXPLAINED, FLAG_JUSTIFIED) else FLAG_UNEXPLAINED
        note_id, llm_reason = str(v.get("matched_note_id", "")).strip(), str(v.get("reason", ""))
        usage = llm.get("usage") or {}
        row.update(model=llm.get("model", ""), source=llm.get("source", ""),
                   input_tokens=usage.get("input_tokens") or 0, output_tokens=usage.get("output_tokens") or 0)
    row.update(llm_flagged=llm_flag, llm_note_id=note_id, llm_reason=llm_reason)

    if llm_flag == FLAG_JUSTIFIED:
        if note_id not in ver:
            row["overridden"] = True
            why = f"the judge cited {note_id or 'no note'}, which was not among the retrieved notes"
        elif not ver[note_id]["ok"]:
            row["overridden"] = True
            why = f"note {note_id} failed verification ({', '.join(ver[note_id]['failed'])})"
        else:
            note = cands[cands["note_id"] == note_id].iloc[0]
            return {**row, "flagged": FLAG_JUSTIFIED, "matched_note_id": note_id,
                    "reason": justified_reason(note, spike)}
        return {**row, "flagged": FLAG_UNEXPLAINED, "matched_note_id": "",
                "reason": (f"Closest note ({_fmt_note(top)}) does not justify the rise: {why}. "
                           f"No genuine justification found; flagged for review.")}

    text = llm_reason.strip().rstrip(".") or "it does not describe a cost rise for this route and week"
    return {**row, "flagged": FLAG_UNEXPLAINED, "matched_note_id": "",
            "reason": f"Closest note ({_fmt_note(top)}): {text}. No genuine justification found; flagged for review."}


def justified_reason(note: pd.Series, spike: pd.Series) -> str:
    """Reason assembled only from the cited note's own text and the computed numbers."""
    text = (f"Matches note {note['note_id']} dated {note['date']:%Y-%m-%d} ({note['applies_to']}): "
            f"\"{note['note'].strip()}\" It describes a cost-raising event covering the week of "
            f"{spike['week_of']:%Y-%m-%d} for this route.")
    if note["applies_to"].lower() == "all routes" and spike.get("flag_basis") == "peers":
        text += (" Caveat: this week was flagged against similar routes, and a cost rise across all "
                 "routes would have lifted those peers too, so the note explains a rise versus the "
                 "route's own history rather than its gap to peers.")
    pct = _PCT_RANGE.search(note["note"])
    vh = spike["overrun_vs_history_pct"]
    if pct and not pd.isna(vh):
        lo, hi = float(pct.group(1)), float(pct.group(2))
        text += (f" The note puts the effect at roughly {lo:g}-{hi:g}%; the observed change is "
                 f"{vh:+.1f}% vs this route's own history"
                 + (", which is more than the note explains." if vh > hi else "."))
    return text


# ----------------------------------------------------------------------------
# Output + contract validation
# ----------------------------------------------------------------------------
def fmt_pct(val: float, suffix: str) -> str:
    return "" if pd.isna(val) else f"{'+' if val >= 0 else ''}{val:.1f}% {suffix}"


def to_submission(detail: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "route": detail["route"],
        "week_of": detail["week_of"].dt.strftime("%Y-%m-%d"),
        "cost_per_tonne_km": detail["cost_per_tonne_km"].round(2),
        "vs_own_history": detail["overrun_vs_history_pct"].map(lambda v: fmt_pct(v, "vs this route's past average")),
        "vs_similar_routes": detail["overrun_vs_peers_pct"].map(lambda v: fmt_pct(v, "vs similar-length routes this week")),
        "flagged": detail["flagged"],
        "matched_note_id": detail["matched_note_id"].fillna(""),
        "reason": detail["reason"],
    })
    return out[CONTRACT_COLUMNS]


def validate_submission(path: Path, sample_path: Path, notes: pd.DataFrame,
                        weekly: pd.DataFrame) -> list[str]:
    """Return a list of contract/grounding violations (empty list = clean)."""
    errs: list[str] = []
    sub = pd.read_csv(path, dtype=str, keep_default_na=False)
    # The provided sample has an unquoted comma in its last row (not strictly valid CSV),
    # so only its header line is used as the column contract.
    sample_cols = sample_path.read_text(encoding="utf-8").splitlines()[0].strip().split(",")
    if list(sub.columns) != sample_cols:
        errs.append(f"columns {list(sub.columns)} != sample {sample_cols}")
        return errs
    if sub.duplicated(["route", "week_of"]).any():
        errs.append("duplicate (route, week_of) rows")
    spikes = weekly[weekly["is_spike"]]
    expected = {(r.route, f"{r.week_of:%Y-%m-%d}") for r in spikes.itertuples()}
    got = set(zip(sub["route"], sub["week_of"]))
    if expected != got:
        errs.append(f"row set mismatch: missing={len(expected - got)} extra={len(got - expected)}")
    note_idx = notes.set_index("note_id", drop=False)
    for _, r in sub.iterrows():
        tag = f"{r['route']} {r['week_of']}"
        if r["flagged"] not in (FLAG_UNEXPLAINED, FLAG_JUSTIFIED):
            errs.append(f"{tag}: invalid flagged value {r['flagged']!r}")
        if not r["reason"].strip():
            errs.append(f"{tag}: empty reason")
        try:
            float(r["cost_per_tonne_km"])
        except ValueError:
            errs.append(f"{tag}: cost_per_tonne_km not numeric")
        if r["flagged"] == FLAG_UNEXPLAINED and r["matched_note_id"]:
            errs.append(f"{tag}: unexplained row must have blank matched_note_id")
        if r["flagged"] == FLAG_JUSTIFIED:
            nid = r["matched_note_id"]
            if nid not in note_idx.index:
                errs.append(f"{tag}: cites unknown note {nid!r}")
                continue
            v = verify_note(note_idx.loc[nid], r["route"], pd.Timestamp(r["week_of"]))
            if not v["ok"]:
                errs.append(f"{tag}: cited {nid} fails {v['failed']}")
            if nid not in r["reason"]:
                errs.append(f"{tag}: reason does not name {nid}")
    return errs


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
def run_pipeline(args) -> int:
    t0 = time.time()
    shipments = load_shipments(DATA_DIR / "shipment_records.csv")
    weekly = build_weekly_table(shipments)
    notes = load_notes(DATA_DIR / "context_notes.csv")
    spikes = weekly[weekly["is_spike"]].copy()
    if args.limit:
        spikes = spikes.head(args.limit)
    print(f"Weeks processed: {len(weekly)} | spikes flagged: {len(spikes)}")

    retriever = Retriever(notes, args.backend)
    # --model pins a single model: no silent fallback to another model between rows or runs,
    # so a verdict can never depend on which model happened to be available that minute.
    chain = [args.model] if args.model else MODEL_CHAIN
    judge = LLMJudge(Path(args.cache_path), use_cache=not args.no_cache, model_chain=chain,
                     max_retries=args.max_retries or (8 if args.model else MAX_RETRIES))
    print(f"Retriever backend: {retriever.backend} | cached verdicts: {len(judge.cache)}")

    rows = []
    for _, spike in spikes.iterrows():
        cands = retriever.candidates(spike["route"], spike["week_of"])
        llm = None if cands.empty else judge.judge(spike, cands)
        res = decide(spike, cands, llm, fallback=args.fallback)
        res["candidates"] = ",".join(cands["note_id"])
        rows.append({**spike.to_dict(), **res})
    detail = pd.DataFrame(rows)
    detail["_sort"] = detail["overrun_vs_history_pct"].fillna(-1e9)
    detail = (detail.sort_values(["_sort", "route", "week_of"], ascending=[False, True, True], kind="mergesort")
                    .drop(columns="_sort").reset_index(drop=True))

    OUT_DIR.mkdir(exist_ok=True)
    errors = detail[detail["status"] == "error"]
    suffix = "_incomplete" if len(errors) else ""   # never overwrite good outputs with a failed run
    detail.to_csv(OUT_DIR / f"audit_detail{suffix}.csv", index=False)
    run_log = {**judge.stats, "retriever": retriever.backend, "spikes": len(detail),
               "rule_rows_no_llm": int((detail["status"] == "rule").sum()),
               "llm_error_rows": len(errors), "seconds": round(time.time() - t0, 1),
               "flagged_counts": detail["flagged"].value_counts().to_dict()}
    (OUT_DIR / f"run_log{suffix}.json").write_text(json.dumps(run_log, indent=2))
    if len(errors):
        print(f"ABORT: {len(errors)} rows had no LLM verdict. Re-run (cache resumes) or use --fallback verifier.")
        return 2

    out_path = Path(args.out)
    to_submission(detail).to_csv(out_path, index=False, lineterminator="\n", encoding="utf-8")
    violations = validate_submission(out_path, DATA_DIR / "sample_output_format_v2.csv", notes, weekly)
    print(json.dumps(run_log, indent=2))
    print(f"Wrote {out_path} ({len(detail)} rows). Contract/grounding violations: {len(violations)}")
    for v in violations:
        print("  -", v)
    return 1 if violations else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the full pipeline")
    r.add_argument("--no-cache", action="store_true", help="ignore cached LLM verdicts (live calls)")
    r.add_argument("--cache-path", default=str(OUT_DIR / "llm_cache.json"))
    r.add_argument("--backend", default="auto", choices=["auto", "minilm", "tfidf"])
    r.add_argument("--fallback", default="abort", choices=["abort", "verifier"],
                   help="what to do when the LLM is unreachable for a row")
    r.add_argument("--model", default="", help="pin ONE model (disables fallback); recommended for repro runs")
    r.add_argument("--max-retries", type=int, default=0, help="retries per model (default 3, or 8 when --model is set)")
    r.add_argument("--limit", type=int, default=0, help="only first N spikes (smoke test)")
    r.add_argument("--out", default=str(OUT_DIR / "final_submission.csv"))
    sub.add_parser("validate", help="validate outputs/final_submission.csv against the contract")
    args = ap.parse_args(argv)

    if args.cmd == "run":
        OUT_DIR.mkdir(exist_ok=True)
        return run_pipeline(args)
    weekly = build_weekly_table(load_shipments(DATA_DIR / "shipment_records.csv"))
    errs = validate_submission(OUT_DIR / "final_submission.csv", DATA_DIR / "sample_output_format_v2.csv",
                               load_notes(DATA_DIR / "context_notes.csv"), weekly)
    print("OK" if not errs else "\n".join(errs))
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
