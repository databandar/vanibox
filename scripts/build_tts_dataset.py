"""Extract a TTS fine-tuning dataset from Vaani's transcribed subset.

The grind that IS the moat: only ~5% of Vaani clips carry transcripts, and
those transcripts have annotation markup that would poison TTS training.
This script streams districts, keeps clean transcribed clips in the target
language, and writes an audiofolder-format dataset:

    <out>/wavs/*.wav            16 kHz mono (what MMS-VITS trains on)
    <out>/metadata.csv          file_name,transcription  (HF audiofolder)

Load with:  load_dataset("audiofolder", data_dir="<out>")

Usage:
  .venv/bin/python scripts/build_tts_dataset.py \
      --language Marathi --minutes 5 --out datasets/tts_marathi \
      --configs Maharashtra_Nagpur Maharashtra_Gondia
(omit --configs to use every district of the states that speak the language)
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vaanibox.curate import score_clip
from vaanibox.dataset import stream_district
from vaanibox.districts import DISTRICT_CONFIGS

LANGUAGE_STATES = {  # where to look when --configs isn't given
    "Marathi": ["Maharashtra"],
    "Hindi": ["Bihar", "UttarPradesh", "Rajasthan", "MadhyaPradesh", "Delhi"],
    "Telugu": ["AndhraPradesh", "Telangana"],
    "Kannada": ["Karnataka"],
    "Bengali": ["WestBengal"],
    "Tamil": ["TamilNadu"],
}

TARGET_SR = 16000  # MMS-VITS sampling rate
MIN_WORDS = 3
MIN_SCORE = 45.0


def clean_transcript(text: str) -> str | None:
    """Strip Vaani annotation markup; None = clip unusable for TTS."""
    if "[" in text:  # [unintelligible]-style markers: unreliable alignment
        return None
    text = re.sub(r"</?noise>", " ", text)
    text = re.sub(r"\{[^}]*\}", " ", text)  # {english} glosses of code-switches
    text = re.sub(r"<[^>]*>", " ", text)  # any other tag
    text = re.sub(r"\s+", " ", text).strip()
    if len(text.split()) < MIN_WORDS:
        return None
    return text


def resample(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return audio
    import librosa

    return librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", required=True)
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--scan-per-district", type=int, default=800,
                    help="max transcribed clips to pull per district")
    args = ap.parse_args()

    configs = args.configs or [
        c for c in DISTRICT_CONFIGS
        if any(c.startswith(s + "_") for s in LANGUAGE_STATES.get(args.language, []))
    ]
    if not configs:
        sys.exit(f"No districts known for language {args.language}; pass --configs.")

    out = Path(args.out)
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    rows, total_s = [], 0.0

    for config in configs:
        if total_s >= args.minutes * 60:
            break
        print(f"scanning {config} …", flush=True)
        kept_here = 0
        for row in stream_district(
            config, limit=args.scan_per_district, min_duration=2.0,
            require_transcript=True, language=args.language,
        ):
            if total_s >= args.minutes * 60:
                break
            text = clean_transcript(row["transcript"])
            if not text:
                continue
            audio = row["audio"]["array"]
            sr = row["audio"]["sampling_rate"]
            sc = score_clip(audio, sr)
            if sc.score < MIN_SCORE or sc.duration > 20:
                continue
            audio = resample(np.asarray(audio, dtype=np.float32), sr)
            name = f"{config}_{row.get('speakerID') or 'x'}_{len(rows):05d}.wav"
            sf.write(out / "wavs" / name, audio, TARGET_SR)
            rows.append({"file_name": f"wavs/{name}", "transcription": text,
                         "speaker_id": str(row.get("speakerID") or ""),
                         "district": config, "score": round(sc.score)})
            total_s += len(audio) / TARGET_SR
            kept_here += 1
        print(f"  kept {kept_here} clips, running total {total_s / 60:.1f} min")

    with open(out / "metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file_name", "transcription", "speaker_id", "district", "score"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n{len(rows)} clips / {total_s / 60:.1f} minutes -> {out}")
    print(f'verify with: load_dataset("audiofolder", data_dir="{out}")')


if __name__ == "__main__":
    main()
