"""Export curated speakers as voice packs.

A voice pack is a folder-per-voice layout that any voice-cloning TTS can use,
with the reference wav front and center. Voicebox's "Add Voice" flow takes a
reference audio sample, so packs import directly: pick the wav, paste the
metadata. Attribution files satisfy Vaani's CC-BY-4.0 license.
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import soundfile as sf

ATTRIBUTION = (
    "Voice reference audio derived from the Vaani dataset "
    "(ARTPARK, IISc Bangalore; Google-funded), "
    "https://huggingface.co/datasets/ARTPARK-IISc/Vaani, "
    "licensed CC-BY-4.0. Exported by VaaniBox on {date}."
)


def export_voice_pack(profiles, out_dir: str | Path, clips_per_voice: int = 2) -> list[Path]:
    """Write each VoiceProfile as a folder with reference wavs + voice.json.

    Returns the list of created voice folders.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    created = []

    for prof in profiles:
        clips = prof.top_clips(clips_per_voice)
        if not clips:
            continue
        name = f"{prof.language or 'Unknown'}_{prof.district or 'Unknown'}_{prof.speaker_id[:12]}"
        vdir = out_dir / name
        vdir.mkdir(parents=True, exist_ok=True)

        wavs = []
        for i, (score, audio, sr, transcript) in enumerate(clips):
            wav_path = vdir / f"reference_{i + 1}.wav"
            sf.write(wav_path, np.asarray(audio, dtype=np.float32), sr)
            wavs.append({"file": wav_path.name, "transcript": transcript, **score.as_dict()})

        meta = {
            "name": name,
            "speaker_id": prof.speaker_id,
            "language": prof.language,
            "gender": prof.gender,
            "state": prof.state,
            "district": prof.district,
            "best_quality_score": round(prof.best_score, 1),
            "references": wavs,
            "source": "ARTPARK-IISc/Vaani (CC-BY-4.0)",
            "voicebox_import": (
                "In Voicebox: Voices -> Add Voice -> select reference_1.wav as the "
                "cloning sample. Works with Chatterbox Multilingual and Qwen3-TTS engines."
            ),
        }
        (vdir / "voice.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        (vdir / "ATTRIBUTION.txt").write_text(ATTRIBUTION.format(date=date.today().isoformat()))
        created.append(vdir)

    return created
