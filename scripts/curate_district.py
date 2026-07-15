"""Add a district's voices to the library from the command line.

This replaces the old studio UI as the way to grow the voice factory:
stream a Vaani district, keep clone-quality clips, save speakers to
vaanibox.db — then run make_district_characters.py to turn the library
into composite RJ/host voices.

Usage:  .venv/bin/python scripts/curate_district.py Maharashtra_Pune [--clips 60]
Needs a Hugging Face token in .hf_token (Vaani terms accepted).
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vaanibox import store
from vaanibox.curate import build_profiles
from vaanibox.dataset import stream_district
from vaanibox.districts import DISTRICT_CONFIGS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="State_District, e.g. Maharashtra_Pune")
    ap.add_argument("--clips", type=int, default=60, help="clips to scan")
    ap.add_argument("--min-score", type=float, default=40.0)
    args = ap.parse_args()

    if args.config not in DISTRICT_CONFIGS:
        hints = [c for c in DISTRICT_CONFIGS if args.config.lower() in c.lower()]
        sys.exit(f"Unknown config {args.config!r}. Close matches: {hints[:5] or DISTRICT_CONFIGS[:5]}")

    print(f"streaming {args.clips} clips from {args.config} …")
    profiles = build_profiles(
        stream_district(args.config, limit=args.clips), min_score=args.min_score
    )
    keyed = {re.sub(r"[^A-Za-z0-9_-]", "_", p.speaker_id): p for p in profiles}
    store.save_profiles(keyed)
    print(f"saved {len(profiles)} speakers to the library:")
    for p in profiles:
        print(f"  {p.speaker_id}  {p.language}  {p.gender or '?'}  score {p.best_score:.0f}")
    print("\nnow run:  .venv/bin/python scripts/make_district_characters.py")


if __name__ == "__main__":
    main()
