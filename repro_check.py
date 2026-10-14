"""Reproducibility check: run the pipeline N times and diff the outputs.

    python repro_check.py                 # cached LLM verdicts (offline, no API key needed)
    python repro_check.py --live          # --no-cache: every LLM call is made again (needs GEMINI_API_KEY)

Each run writes its own CSV and its own copy of the verdict cache, so runs cannot influence
one another. The report states which mode was used: the cached mode demonstrates that every
deterministic stage (weekly numbers, flags, retrieval, verifier, reasons, ordering, CSV
formatting) is byte-identical, but it cannot show that the LLM itself is stable. --live does.
"""
import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
STABLE_COLS = ["route", "week_of", "cost_per_tonne_km", "vs_own_history", "vs_similar_routes",
               "flagged", "matched_note_id"]          # the fields the brief says must not move


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--backend", default="auto", choices=["auto", "minilm", "tfidf"])
    ap.add_argument("--model", default="", help="pin one Gemini model for every run (recommended with --live)")
    args = ap.parse_args()

    work = OUT / "repro"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    paths = []
    for i in range(1, args.runs + 1):
        out = work / f"run{i}.csv"
        cache = work / f"cache{i}.json"
        if not args.live and (OUT / "llm_cache.json").exists():
            shutil.copy(OUT / "llm_cache.json", cache)
        cmd = [sys.executable, str(ROOT / "freight_audit.py"), "run", "--backend", args.backend,
               "--out", str(out), "--cache-path", str(cache)] + (["--no-cache"] if args.live else []) + (["--model", args.model] if args.model else [])
        rc = subprocess.run(cmd, capture_output=True, text=True).returncode
        print(f"run {i}: exit={rc} -> {out.name}")
        if rc != 0:
            print("run failed; see freight_audit.py output")
            return 1
        paths.append(out)

    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    frames = [pd.read_csv(p, keep_default_na=False) for p in paths]
    stable_equal = all(frames[0][STABLE_COLS].equals(f[STABLE_COLS]) for f in frames[1:])
    full_equal = len(set(hashes)) == 1
    lines = [f"mode: {'LIVE LLM calls (--no-cache)' if args.live else 'cached LLM verdicts (deterministic stages only)'}",
             f"model pinned: {args.model or 'no (fallback chain)'}",
             f"runs: {args.runs}", *(f"sha256 run{i + 1}: {h}" for i, h in enumerate(hashes)),
             f"verdict/cited-note/number columns identical across runs: {stable_equal}",
             f"entire CSV (including reason text) byte-identical: {full_equal}"]
    if not full_equal:
        lines.append("(reason wording differs between runs; allowed by the brief if the columns above match)")
    (OUT / "repro_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if stable_equal else 1


if __name__ == "__main__":
    sys.exit(main())
