"""Generate synthetic 'speech-like' clips so the whole pipeline can be tried
without Hugging Face access. Each fake speaker gets a distinct pitch/cadence;
clips vary in quality (some too short, too quiet, or noisy) so curation has
something real to reject."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SR = 16000
rng = np.random.default_rng(7)


def speechish(duration, f0, rate, level, noise=0.005):
    """Amplitude-modulated harmonic tone: crude but has speech-like envelope
    statistics (syllable bursts + pauses), enough to exercise the scorer."""
    t = np.arange(int(duration * SR)) / SR
    carrier = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in (1, 2, 3))
    syllables = (np.sin(2 * np.pi * rate * t) > -0.2).astype(float)
    pauses = (np.sin(2 * np.pi * 0.25 * t + rng.uniform(0, 6)) > -0.6).astype(float)
    env = syllables * pauses
    env = np.convolve(env, np.ones(800) / 800, mode="same")
    sig = level * carrier / 3 * env + rng.normal(0, noise, len(t))
    return sig.astype(np.float32)


def rows():
    speakers = [
        ("VB_DEMO_001", "Hindi", "Female", 210, 4.0),
        ("VB_DEMO_002", "Hindi", "Male", 120, 3.2),
        ("VB_DEMO_003", "Magahi", "Female", 190, 4.5),
    ]
    out = []
    for spk, lang, gender, f0, rate in speakers:
        specs = [
            (12.0, 0.28, 0.005),   # good
            (8.0, 0.22, 0.008),    # good
            (1.5, 0.30, 0.005),    # too short -> rejected
            (10.0, 0.008, 0.004),  # too quiet -> rejected
            (9.0, 0.25, 0.15),     # very noisy -> low score
        ]
        for dur, level, noise in specs:
            out.append(
                {
                    "audio": {"array": speechish(dur, f0, rate, level, noise), "sampling_rate": SR},
                    "speakerID": spk,
                    "language": lang,
                    "gender": gender,
                    "state": "DemoState",
                    "district": "DemoDistrict",
                    "duration": dur,
                    "transcript": "(synthetic demo clip)",
                }
            )
    return out


if __name__ == "__main__":
    from vaanibox.curate import build_profiles
    from vaanibox.export import export_voice_pack

    profiles = build_profiles(rows())
    print(f"{len(profiles)} demo profiles curated:")
    for p in profiles:
        print(f"  {p.speaker_id} ({p.language}, {p.gender}): {len(p.clips)} clips kept, best score {p.best_score:.0f}")
    created = export_voice_pack(profiles, Path(__file__).resolve().parents[1] / "voice_packs" / "demo")
    print(f"exported {len(created)} voice packs -> {created[0].parent}")
