"""Build consent-safe district voices from the curated Vaani library.

Each district character is a COMPOSITE: the best clips of up to three
different speakers (same gender when possible) concatenated into one target
reference. The voice-conversion embedding averages across them, so the result
carries the district's accent without being any single real person's voice.

Run after curating districts in the studio:
    .venv/bin/python scripts/make_district_characters.py
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vaanibox.store import load_profiles

OUT = ROOT / "characters"
MAX_SPEAKERS = 3
MAX_SECONDS = 24.0


def main():
    profiles = load_profiles()
    if not profiles:
        print("Voice library is empty — curate districts in the studio first.")
        return

    by_district = defaultdict(list)
    for p in profiles.values():
        if p.district:
            by_district[(p.state, p.district)].append(p)

    OUT.mkdir(exist_ok=True)
    for (state, district), profs in sorted(by_district.items()):
        # prefer a single-gender blend so the composite sounds coherent
        by_gender = defaultdict(list)
        for p in profs:
            by_gender[p.gender or "?"].append(p)
        gender, group = max(by_gender.items(), key=lambda kv: len(kv[1]))
        group = sorted(group, key=lambda p: -p.best_score)[:MAX_SPEAKERS]

        pieces, total, sr_out = [], 0.0, None
        for p in group:
            score, audio, sr, _ = p.top_clips(1)[0]
            sr_out = sr_out or sr
            if sr != sr_out:
                continue
            take = min(len(audio) / sr, MAX_SECONDS / len(group))
            pieces.append(np.asarray(audio[: int(take * sr)], dtype=np.float32))
            total += take
        if not pieces:
            continue
        blend = np.concatenate(pieces)
        name = f"{district.lower()}_{state.lower()}"
        sf.write(OUT / f"{name}.wav", blend, sr_out)
        print(f"{name}: {len(group)} speaker(s), {gender}, {total:.0f}s composite")

    print(f"\nDistrict characters written to {OUT}/ — bot and web app list them automatically.")


if __name__ == "__main__":
    main()
