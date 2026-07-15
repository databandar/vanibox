# 📻 Radio Vaani

**An AI radio station and fake podcast in real Indian district voices — for people who ship.**

A late-night coding companion that runs entirely on your machine. The RJ and podcast
hosts speak in **composite voices built from real speakers in the
[Vaani dataset](https://huggingface.co/datasets/ARTPARK-IISc/Vaani)** (ARTPARK/IISc,
CC-BY-4.0) — a blend of 2–3 speakers per district, so every voice carries an authentic
regional accent while being no single real person. No celebrity cloning, ever.

Unlike cloud AI-radio products (radio69.ai, airadiobot, NotebookLM-style podcast
generators): local-first, Indian-accented, developer-targeted, free.

## Run it

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/build_radio_demo.py   # pre-render RJ, podcast, lo-fi (~10 min first run)
.venv/bin/python app.py                        # → http://127.0.0.1:5051
```

Two channels:
- **📻 Radio** — RJ Vaani (Nagpur composite) does Hinglish patter, focus-block calls,
  and song handoffs between tracks, looping forever.
- **🎙 Podcast** — *Stack Underflow*: Arjun (Koriya composite) and Meera (Araria
  composite) on why your code only works at 2 AM.

## Your music

Drop `.mp3`/`.wav` files into `radio/music/` and re-run `scripts/build_radio_demo.py` —
every track enters the rotation with RJ handoffs between them. The bundled procedural
lo-fi is just placeholder. (Use music you own or that is openly licensed.)

## Growing the voice cast

```bash
.venv/bin/python scripts/curate_district.py Maharashtra_Pune   # stream + curate a district
.venv/bin/python scripts/make_district_characters.py           # rebuild composite voices
```

Curation streams the gated Vaani dataset (accept the terms on Hugging Face, put a read
token in `.hf_token`), scores clips for cloning quality, and stores speakers in a local
SQLite library. Composites blend up to three same-gender speakers per district.

## Layout

```
app.py                              Flask server (radio page + audio)
templates/radio.html                the on-air console UI
scripts/build_radio_demo.py         pre-renders RJ, podcast, music, playlist.json
scripts/curate_district.py          add a district's speakers to the library
scripts/make_district_characters.py library -> composite character voices
vaanibox/                           voice factory: Vaani streaming, clip scoring,
                                    SQLite library, Chatterbox TTS wrapper
```

Earlier incarnations of this repo (voice-cloning studio, Telegram voice-changer bot,
Accent Atlas maps) live in git history before the radio pivot.

## License & ethics

Code MIT. Vaani audio CC-BY-4.0 (attribution in exported audio metadata). Voices are
composites of consenting dataset speakers — never impersonations of identifiable
people. Generated speech is watermarked by Chatterbox's built-in Perth watermarker.
