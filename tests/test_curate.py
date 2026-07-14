import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.make_demo_data import rows, speechish, SR
from vaanibox.curate import build_profiles, score_clip


def test_good_clip_scores_high():
    sc = score_clip(speechish(12.0, 200, 4.0, 0.28), SR)
    assert sc.score > 60, sc


def test_short_clip_rejected():
    sc = score_clip(speechish(1.5, 200, 4.0, 0.28), SR)
    assert sc.score < 40, sc


def test_silence_scores_low():
    sc = score_clip(np.zeros(SR * 10, dtype=np.float32), SR)
    assert sc.score < 40, sc


def test_quiet_clip_penalized():
    loud = score_clip(speechish(10.0, 200, 4.0, 0.28), SR)
    quiet = score_clip(speechish(10.0, 200, 4.0, 0.008), SR)
    assert quiet.score < loud.score, (quiet, loud)


def test_build_profiles_groups_and_filters():
    profiles = build_profiles(rows(), min_score=40)
    assert len(profiles) == 3, [p.speaker_id for p in profiles]
    for p in profiles:
        # the too-short and too-quiet clips must have been dropped
        assert 1 <= len(p.clips) <= 4, (p.speaker_id, len(p.clips))
        assert p.best_score > 50


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
