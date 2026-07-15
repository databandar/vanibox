# 📻 Radio Vaani

**Your music, with an AI RJ — a late-night radio station for people who ship.**

Drop your MP3s in a folder; Meera, the AI RJ, introduces every song by name,
calls focus blocks, and keeps you company between tracks. A second channel has
*Stack Underflow*, a fake podcast where two AI hosts discuss why your code only
works at 2 AM. Everything is generated and served on your machine — no cloud,
no accounts, no subscriptions.

## Run it

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/build_radio_demo.py   # render RJ + podcast (~2 min)
.venv/bin/python app.py                        # → http://127.0.0.1:5051
```

## Your music

Put files in `radio/music/` named `Artist - Title.mp3` and re-run the
generator — the RJ announces each one on air ("Up next — Jai Ho, by
A. R. Rahman"). Use music you own or that is openly licensed. If the folder is
empty, procedural lo-fi filler is synthesized so the station still runs.

## Voices

All voices are Kokoro (82M, local): **Meera** (`af_heart`) hosts the radio and
co-hosts the podcast with **Dev** (`am_michael`). Edit the lines at the top of
`scripts/build_radio_demo.py` to change what they say; `--podcast-only` skips
re-rendering the radio.

## Layout

```
app.py                        Flask server (radio page + audio)
templates/radio.html          the on-air console UI
scripts/build_radio_demo.py   renders RJ, song intros, podcast, playlist.json
radio/                        generated audio + your music (gitignored)
```

This repo previously housed VaaniBox (a Vaani-dataset voice-cloning studio,
Telegram voice bot, and accent atlas) — all of it lives in git history before
the radio pivot.
