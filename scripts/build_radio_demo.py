"""Pre-render Radio Vaani: RJ segments, song intros, and the podcast.

Everything is Kokoro (local, fast, no accounts): Meera (af_heart) is the RJ
and podcast co-host, Dev (am_michael) co-hosts the podcast. Songs in
radio/music/ named "Artist - Title.mp3" get real on-air introductions.

Target listener: developers coding late.

Run:  .venv/bin/python scripts/build_radio_demo.py [--podcast-only]
Add/remove music in radio/music/ and re-run to refresh the rotation.
If the music folder is empty, three procedural lo-fi loops are synthesized
as filler.
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

RJ_VOICE = "af_heart"  # Meera

STATION_LINES = {
    "intro": "You're listening to Radio Vaani. Music for people who ship. Your build is running, the night is young... stay with me.",
    "focus": "Focus block, people. Twenty five minutes. No new tabs, no chai run, just you and the cursor. I'll be back when it's done.",
    "trivia": "Fun fact: everything you hear between the songs is generated on this very laptop. I'm not real. The music, thankfully, is.",
    "night": "It's two A M and you're still coding. Respect. One more song, then please consider sleeping.",
}

SONG_INTRO_TEMPLATES = [
    "Up next — {title}, by {artist}.",
    "Here's {artist}, with {title}. Back to work.",
    "This one is {title}. {artist}. Don't you dare skip it.",
    "And now, {artist}. {title}. You know this one.",
    "Keeping you company: {title}, {artist}.",
]

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

_pipe = None


def say(text: str, voice: str) -> np.ndarray:
    global _pipe
    if _pipe is None:
        from kokoro import KPipeline

        _pipe = KPipeline(lang_code="a")
    chunks = [np.asarray(a, dtype=np.float32) for _, _, a in _pipe(text, voice=voice)]
    return np.concatenate(chunks)


# ---------------- procedural lo-fi (filler when music/ is empty) ----------------

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
        if b % 2 == 0:
            t = np.arange(int(0.12 * SR)) / SR
            kick = 0.5 * np.sin(2 * np.pi * (60 - 120 * t) * t) * np.exp(-t * 30)
            out[at:at + len(kick)] += kick.astype(np.float32)
        tick = rng.normal(0, 0.05, int(0.03 * SR)).astype(np.float32) * np.exp(-np.arange(int(0.03 * SR)) / (0.004 * SR))
        out[at:at + len(tick)] += tick
    return out


PROGRESSIONS = [
    [[174.6, 220.0, 261.6, 329.6], [164.8, 196.0, 246.9, 293.7], [146.8, 174.6, 220.0, 261.6], [130.8, 164.8, 196.0, 246.9]],
    [[146.8, 174.6, 220.0], [130.8, 164.8, 196.0], [123.5, 146.8, 185.0], [110.0, 130.8, 164.8]],
    [[196.0, 246.9, 293.7], [174.6, 220.0, 261.6], [164.8, 207.7, 246.9], [146.8, 185.0, 220.0]],
]


def make_track(prog, minutes=1.1, bpm=72):
    spb = 60 / bpm
    bar = 4 * spb
    chords = []
    for _ in range(max(1, int((minutes * 60) / (len(prog) * bar)))):
        for c in prog:
            chords.append(_chord(c, bar))
    music = np.concatenate(chords)
    drums = _drums(int(len(music) / SR / spb), spb)[: len(music)]
    crackle = np.zeros(len(music), dtype=np.float32)
    pops = rng.choice(len(music), size=max(4, len(music) // 24000), replace=False)
    crackle[pops] = rng.uniform(-0.12, 0.12, len(pops))
    mix = music + drums + crackle + rng.normal(0, 0.004, len(music)).astype(np.float32)
    mix = np.convolve(mix, np.ones(9) / 9, mode="same")
    mix = mix / (np.max(np.abs(mix)) + 1e-6) * 0.7
    fade = int(1.5 * SR)
    mix[:fade] *= np.linspace(0, 1, fade)
    mix[-fade:] *= np.linspace(1, 0, fade)
    return mix.astype(np.float32)


# ---------------- content ----------------

def music_library() -> list[str]:
    return sorted(
        p.name for p in (RADIO / "music").iterdir()
        if p.suffix.lower() in (".wav", ".mp3", ".m4a") and not p.name.startswith("lofi_")
    )


def make_rj(tracks: list[str]):
    print("RJ segments (Meera)…")
    for name, line in STATION_LINES.items():
        audio = say(line, RJ_VOICE)
        sf.write(RADIO / "rj" / f"{name}.wav", audio, SR)
        print(f"  rj/{name}: {len(audio)/SR:.1f}s")
    for i, track in enumerate(tracks):
        stem = track.rsplit(".", 1)[0]
        artist, _, title = stem.partition(" - ")
        if not title:
            artist, title = "", stem
        line = SONG_INTRO_TEMPLATES[i % len(SONG_INTRO_TEMPLATES)].format(
            artist=artist.strip() or "someone you know", title=title.strip().replace("_", " ")
        )
        audio = say(line, RJ_VOICE)
        sf.write(RADIO / "rj" / f"song_{i:02d}.wav", audio, SR)
        print(f"  rj/song_{i:02d}: “{line}”")


def make_podcast():
    print("podcast…")
    gap = np.zeros(int(0.45 * SR), dtype=np.float32)
    pieces = []
    for voice, line in PODCAST["turns"]:
        preset = {"michael": "am_michael", "heart": "af_heart"}[voice]
        pieces.append(say(line, preset))
        pieces.append(gap)
    audio = np.concatenate(pieces)
    audio = audio / (np.max(np.abs(audio)) + 1e-6) * 0.85
    sf.write(RADIO / "podcast" / "stack_underflow_ep1.wav", audio, SR)
    print(f"  podcast: {len(audio)/SR:.0f}s")
    return len(audio) / SR


def build_playlist(tracks: list[str], pod_dur: float):
    station = ["focus", "trivia", "night"]
    rotation = [{"type": "rj", "file": "rj/intro.wav", "label": "RJ Meera · Radio Vaani"}]
    for i, track in enumerate(tracks):
        if i and i % 3 == 0:  # a station bit every few songs
            bit = station[(i // 3 - 1) % len(station)]
            rotation.append({"type": "rj", "file": f"rj/{bit}.wav", "label": "RJ Meera"})
        rotation.append({"type": "rj", "file": f"rj/song_{i:02d}.wav", "label": "RJ Meera"})
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
    print(f"playlist.json: {len(rotation)} radio segments, {len(tracks)} songs, 1 podcast")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--podcast-only", action="store_true")
    args = ap.parse_args()

    for d in ["music", "rj", "podcast"]:
        (RADIO / d).mkdir(parents=True, exist_ok=True)

    tracks = music_library()
    if not tracks:  # nothing to play — synthesize filler
        print("music folder empty — synthesizing lo-fi filler…")
        for i, prog in enumerate(PROGRESSIONS, 1):
            sf.write(RADIO / "music" / f"lofi_{i}.wav", make_track(prog), SR)
        tracks = sorted(p.name for p in (RADIO / "music").iterdir() if p.suffix == ".wav")

    if not args.podcast_only:
        make_rj(tracks)
    pod_dur = make_podcast()
    build_playlist(tracks, pod_dur)


if __name__ == "__main__":
    main()
