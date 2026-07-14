# 🇮🇳 VaaniBox

**Every district of India, as a voice you can use.**

VaaniBox bridges two open projects:

- **[Vaani](https://huggingface.co/datasets/ARTPARK-IISc/Vaani)** (ARTPARK / IISc Bangalore) — 31,000+ hours of spontaneous speech from **156K speakers across 165 districts and 106 languages**, CC-BY-4.0.
- **[Voicebox](https://github.com/jamiepine/voicebox)** (jamiepine) — the open-source, local-first AI voice studio with zero-shot voice cloning (Chatterbox, Qwen3-TTS, and more), MIT.

Vaani has the voices; Voicebox has the cloning engines. VaaniBox streams Vaani, **auto-curates clone-quality reference clips** (duration, loudness, SNR, speech-ratio, clipping heuristics — no model downloads needed), and **exports Voicebox-ready voice packs**. The result: text-to-speech in authentic regional Indian accents at *district-level* granularity — something no commercial TTS offers.

## Quickstart

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python app.py          # Flask studio at http://127.0.0.1:5050
```

(The original Gradio UI still works: `.venv/bin/python gradio_app.py` → port 7860.)

**No Hugging Face account?** Click **"Demo mode"** in the app — it runs the full curate → preview → export pipeline on synthetic voices.

**With real Vaani data:** accept the terms at [the dataset page](https://huggingface.co/datasets/ARTPARK-IISc/Vaani) (free, auto-approved), then either run `hf auth login` or put a read token in a `.hf_token` file in the project root (gitignored). Pick any state/district in the app — audio is streamed, nothing near the 3.6 TB dataset ever hits your disk.

## What the app does

1. **Explore & curate** — pick a state and district (165 available), stream N clips, and get a ranked table of speakers whose clips pass cloning-quality thresholds. Click a row to listen and select it.
2. **Speak in this voice** — type any text (Hindi, Bengali, Tamil, … or English) and generate it in the selected speaker's voice, entirely in the browser. Cloning runs locally on Chatterbox Multilingual (PyTorch on Apple-Silicon MPS); the ~3 GB model downloads once on first generate.
3. **Export voice packs (optional)** — writes `voice_packs/<Language_District_Speaker>/` folders containing the best reference wavs, `voice.json` metadata, and CC-BY attribution, for the Voicebox desktop app or any other cloning TTS.

## Project layout

```
app.py                    Flask studio (explore / export / clone API + web UI)
templates/index.html      The studio front-end (vanilla JS)
gradio_app.py             Legacy Gradio version of the same studio
vaanibox/districts.py     All 165 State_District configs (auto-generated from HF API)
vaanibox/dataset.py       Gated streaming access to Vaani
vaanibox/curate.py        Clip quality scoring + per-speaker profiling (pure numpy)
vaanibox/export.py        Voice pack writer w/ CC-BY attribution
vaanibox/tts.py           Optional Chatterbox cloning preview
scripts/make_demo_data.py Synthetic speakers for offline demo
tests/test_curate.py      Scorer sanity tests (.venv/bin/python tests/test_curate.py)
```

## Where this can go

- **Accent Atlas** — clickable map of India; hear each district speak.
- **Fine-tuned Indian TTS engine** — use Vaani's 2,043 transcribed hours to fine-tune Chatterbox/Qwen3-TTS and contribute an "Indian languages" engine back to Voicebox.
- **Dialect-aware dictation** — fine-tune Whisper on Vaani to fix Voicebox dictation for Indian-accent English and code-switched speech.
- **Language preservation** — many of Vaani's 106 languages (Mundari, Kurukh, Bhili…) have *zero* TTS support today; curated packs give them a synthetic voice for the first time.

## Licensing

Vaani audio is CC-BY-4.0 — exported packs include `ATTRIBUTION.txt`. Voicebox is MIT. **Ethics note:** these are real people's voices; use packs for accent-authentic narration, research, and accessibility — not impersonation. Voice packs are for cloning *accent and texture into new speech*, and Vaani speakers consented to research/commercial dataset use, but be thoughtful.
