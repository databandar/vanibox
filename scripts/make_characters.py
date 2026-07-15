"""Seed characters/ with Voicebox's default preset voices.

Voicebox ships the Kokoro engine (82M) with named preset voices; this script
renders a short reference clip for each preset into characters/, where the
Telegram bot (and ChatterboxVC) picks them up as conversion targets.

Run once:  .venv/bin/python scripts/make_characters.py
Re-run any time — it overwrites the same files.
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "characters"

# Kokoro preset -> character file name. a=American, b=British; f/m = voice.
VOICES = {
    "af_heart": "heart",
    "af_bella": "bella",
    "am_adam": "adam",
    "am_michael": "michael",
    "bf_emma": "emma",
    "bm_george": "george",
}

TEXT = (
    "Hello there! This is my voice, and I rather like it. "
    "I can read your messages, tell your stories, and say almost anything "
    "you want, in my own particular way of speaking."
)


def main():
    from kokoro import KPipeline

    OUT.mkdir(exist_ok=True)
    pipeline_us = KPipeline(lang_code="a")  # American English presets
    pipeline_gb = KPipeline(lang_code="b")  # British English presets

    for preset, name in VOICES.items():
        pipe = pipeline_gb if preset.startswith("b") else pipeline_us
        chunks = [audio for _, _, audio in pipe(TEXT, voice=preset)]
        audio = np.concatenate([np.asarray(c, dtype=np.float32) for c in chunks])
        path = OUT / f"{name}.wav"
        sf.write(path, audio, 24000)
        print(f"{name:8} <- {preset}: {len(audio) / 24000:.1f}s -> {path.name}")

    print(f"\n{len(VOICES)} Voicebox default characters in {OUT}/ — the bot lists them automatically.")


if __name__ == "__main__":
    main()
