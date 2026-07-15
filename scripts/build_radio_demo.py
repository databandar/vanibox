"""Pre-render the Radio Vaani demo: lo-fi music, RJ segments in a district
composite voice, and a two-host fake podcast episode.

Target listener: developers coding late, in India. The RJ speaks Hinglish-
flavored English (plus one Hindi drop) in the Nagpur composite voice — a
blend of real Vaani speakers, no single real person. Podcast hosts are two
distinct preset voices.

Output -> radio/  (music/*.wav, rj/*.wav, podcast/*.wav, playlist.json)
Run:  .venv/bin/python scripts/build_radio_demo.py
Drop your own .mp3/.wav into radio/music/ any time — the playlist builder
picks them up on next run.
"""

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RADIO = ROOT / "radio"
SR = 24000
rng = np.random.default_rng(11)

RJ_REF = ROOT / "characters" / "nagpur_maharashtra.wav"  # composite, not a person

RJ_LINES = [
    ("intro", "en", "You're listening to Radio Vaani. Voices from a hundred and sixty five districts, music for people who ship. Your build is running, the night is young... stay with me."),
    ("handoff1", "en", "Okay, enough talk. Here's something slow from the VaaniBox lab. Your bug will still be there after this track. Sadly."),
    ("focus", "en", "Focus block, people. Twenty five minutes. No new tabs, no chai run, just you and the cursor. I'll be back when it's done. Chalo, shuru karo."),
    ("trivia", "en", "Fun fact. This voice you're hearing? It's a blend of three real speakers from Nagpur, Maharashtra. No single person sounds exactly like me. I am, quite literally, everyone."),
    ("night", "hi", "रात के दो बजे हैं और आप अभी भी कोड कर रहे हैं। वाह। एक और गाना हो जाए?"),
    ("handoff2", "en", "That compile error is not going to fix itself. But you know what helps? Lo fi. Probably. Here we go."),
]

# Podcast hosts: Kokoro presets (the Dev & Meera voices) — the RJ keeps the
# Indian district composite; the podcast doesn't have to.
PODCAST = {
    "title": "Stack Underflow — Ep 1: Why your code only works at 2 AM",
    "hosts": "Dev & Meera (AI)",
    "turns": [
        ("michael", "Welcome back to Stack Underflow, the podcast where two AI hosts pretend to understand your codebase. I'm Dev."),
        ("heart", "And I'm Meera. Today's topic: why does code magically work at two in the morning?"),
        ("michael", "Simple. At 2 AM there's nobody left awake to open Slack. Zero interruptions, infinite flow state."),
        ("heart", "My theory is different. By 2 AM you've lowered your standards enough to accept the fix you rejected at 2 PM."),
        ("michael", "Harsh. True, but harsh. Speaking of acceptance — have you tried explaining your bug to a rubber duck?"),
        ("heart", "I don't need a duck, I have you. Same energy, slightly better vocabulary."),
        ("michael", "Fair. Quick listener note — this entire episode was generated on a laptop in one take, voices and all. No cloud, no studio."),
        ("heart", "Which explains the budget. Anyway — chai or coffee for the 2 AM push? I'm team chai, obviously."),
        ("michael", "Coffee. But since this show is recorded in India, I already know I've lost this vote."),
        ("heart", "You have. That's all for episode one — keep shipping, and if it works at 2 AM, don't ask why. Bye!"),
    ],
}


# ---------------- procedural lo-fi ----------------

def _note(freq, dur, level=0.22):
    t = np.arange(int(dur * SR)) / SR
    env = np.minimum(t / 0.4, 1.0) * np.exp(-t / (dur * 0.9))
    wave = sum(np.sin(2 * np.pi * freq * k * t) * a for k, a in [(1, 1.0), (2, 0.35), (3, 0.12)])
    return (level * env * wave / 1.5).astype(np.float32)


def _chord(freqs, dur):
    out = np.zeros(int(dur * SR), dtype=np.float32)
    for f in freqs:
        out += _note(f, dur, level=0.16)
    return out


def _drums(n_beats, spb):
    out = np.zeros(int(n_beats * spb * SR), dtype=np.float32)
    for b in range(n_beats):
        at = int(b * spb * SR)
        if b % 2 == 0:  # soft kick
            t = np.arange(int(0.12 * SR)) / SR
            kick = 0.5 * np.sin(2 * np.pi * (60 - 120 * t) * t) * np.exp(-t * 30)
            out[at:at + len(kick)] += kick.astype(np.float32)
        # hat: filtered noise tick on every beat
        tick = rng.normal(0, 0.05, int(0.03 * SR)).astype(np.float32) * np.exp(-np.arange(int(0.03 * SR)) / (0.004 * SR))
        out[at:at + len(tick)] += tick
    return out


def _vinyl(n):
    crackle = np.zeros(n, dtype=np.float32)
    pops = rng.choice(n, size=max(4, n // 24000), replace=False)
    crackle[pops] = rng.uniform(-0.12, 0.12, len(pops))
    hiss = rng.normal(0, 0.004, n).astype(np.float32)
    return crackle + hiss


PROGRESSIONS = [
    [[174.6, 220.0, 261.6, 329.6], [164.8, 196.0, 246.9, 293.7], [146.8, 174.6, 220.0, 261.6], [130.8, 164.8, 196.0, 246.9]],
    [[146.8, 174.6, 220.0], [130.8, 164.8, 196.0], [123.5, 146.8, 185.0], [110.0, 130.8, 164.8]],
    [[196.0, 246.9, 293.7], [174.6, 220.0, 261.6], [164.8, 207.7, 246.9], [146.8, 185.0, 220.0]],
]


def make_track(prog, minutes=1.1, bpm=72):
    spb = 60 / bpm
    bar = 4 * spb
    chords = []
    n_loops = max(1, int((minutes * 60) / (len(prog) * bar)))
    for _ in range(n_loops):
        for c in prog:
            chords.append(_chord(c, bar))
    music = np.concatenate(chords)
    drums = _drums(int(len(music) / SR / spb), spb)[: len(music)]
    mix = music + drums + _vinyl(len(music))
    # gentle low-pass via cumulative smoothing for the lo-fi warmth
    kernel = np.ones(9) / 9
    mix = np.convolve(mix, kernel, mode="same")
    mix = mix / (np.max(np.abs(mix)) + 1e-6) * 0.7
    # fade edges
    fade = int(1.5 * SR)
    mix[:fade] *= np.linspace(0, 1, fade)
    mix[-fade:] *= np.linspace(1, 0, fade)
    return mix.astype(np.float32)


# ---------------- speech ----------------

def make_rj():
    from vaanibox import tts

    out = []
    for name, lang, line in RJ_LINES:
        sr, audio = tts.clone_speak(line, str(RJ_REF), language_id=lang)
        audio = np.asarray(audio, dtype=np.float32)
        if sr != SR:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        path = RADIO / "rj" / f"{name}.wav"
        sf.write(path, audio, SR)
        out.append((name, len(audio) / SR))
        print(f"  rj/{name}: {len(audio)/SR:.1f}s")
    return out


def make_podcast():
    from kokoro import KPipeline

    pipe = KPipeline(lang_code="a")
    gap = np.zeros(int(0.45 * SR), dtype=np.float32)
    pieces = []
    for voice, line in PODCAST["turns"]:
        preset = {"michael": "am_michael", "heart": "af_heart"}[voice]
        chunks = [np.asarray(a, dtype=np.float32) for _, _, a in pipe(line, voice=preset)]
        pieces.append(np.concatenate(chunks))
        pieces.append(gap)
    audio = np.concatenate(pieces)
    audio = audio / (np.max(np.abs(audio)) + 1e-6) * 0.85
    path = RADIO / "podcast" / "stack_underflow_ep1.wav"
    sf.write(path, audio, SR)
    print(f"  podcast: {len(audio)/SR:.0f}s -> {path.name}")
    return len(audio) / SR


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--podcast-only", action="store_true",
                    help="regenerate only the podcast + playlist (keep music and RJ)")
    args = ap.parse_args()

    for d in ["music", "rj", "podcast"]:
        (RADIO / d).mkdir(parents=True, exist_ok=True)

    if not args.podcast_only:
        print("music…")
        for i, prog in enumerate(PROGRESSIONS, 1):
            track = make_track(prog)
            sf.write(RADIO / "music" / f"lofi_{i}.wav", track, SR)
            print(f"  music/lofi_{i}: {len(track)/SR:.0f}s")

        print("RJ segments (district composite voice)…")
        make_rj()

    print("podcast…")
    pod_dur = make_podcast()

    music_files = sorted(
        p.name for p in (RADIO / "music").iterdir() if p.suffix.lower() in (".wav", ".mp3", ".m4a")
    )
    # radio rotation: alternate RJ bits with every track in music/ — drop your
    # own files there and re-run; more music than RJ lines just recycles the RJ.
    rj_order = ["intro", "handoff1", "focus", "trivia", "night", "handoff2"]
    rotation = []
    for i in range(max(len(rj_order), len(music_files) or 1)):
        rotation.append({"type": "rj", "file": f"rj/{rj_order[i % len(rj_order)]}.wav",
                         "label": "RJ Vaani · Nagpur composite"})
        if music_files:
            track = music_files[i % len(music_files)]
            rotation.append({"type": "music", "file": f"music/{track}",
                             "label": track.rsplit(".", 1)[0].replace("_", " ")})

    playlist = {
        "radio": rotation,
        "podcasts": [{
            "title": PODCAST["title"],
            "file": "podcast/stack_underflow_ep1.wav",
            "duration": round(pod_dur),
            "hosts": PODCAST["hosts"],
        }],
    }
    (RADIO / "playlist.json").write_text(json.dumps(playlist, indent=2))
    print(f"\nplaylist.json written — {len(rotation)} radio segments, 1 podcast")


if __name__ == "__main__":
    main()
