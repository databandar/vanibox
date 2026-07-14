"""VaaniBox — explore Vaani's 165 districts, curate clone-ready voices,
export Voicebox voice packs.

Run:  .venv/bin/python gradio_app.py
"""



import sys
from pathlib import Path

import gradio as gr


sys.path.insert(0, str(Path(__file__).resolve().parent))

from vaanibox.curate import build_profiles
from vaanibox.dataset import VaaniAccessError, has_hf_token, stream_district
from vaanibox.districts import STATES, districts_for
from vaanibox.export import export_voice_pack

PACKS_DIR = Path(__file__).resolve().parent / "voice_packs"

# Session state: profiles from the last curation run, keyed by display label.
_last_profiles: dict[str, object] = {}


def _profiles_to_table(profiles):
    rows = []
    for p in profiles:
        best = p.top_clips(1)[0][0]
        rows.append(
            [
                p.speaker_id,
                p.language,
                p.gender,
                len(p.clips),
                round(p.best_score),
                f"{best.duration:.1f}s",
                f"{best.snr_db:.0f} dB",
            ]
        )
    return rows


def curate_district(state, district, n_samples, min_score, progress=gr.Progress()):
    global _last_profiles
    config = f"{state}_{district}"
    progress(0.05, desc=f"Streaming {config} from Hugging Face…")

    def sample_iter():
        for i, row in enumerate(stream_district(config, limit=int(n_samples))):
            progress(0.05 + 0.9 * i / n_samples, desc=f"Scoring clip {i + 1}/{int(n_samples)}")
            yield row

    try:
        profiles = build_profiles(sample_iter(), min_score=min_score)
    except VaaniAccessError as e:
        return [], gr.update(choices=[]), None, f"⚠️ {e}"

    _last_profiles = {f"{p.speaker_id} · {p.language} · score {p.best_score:.0f}": p for p in profiles}
    status = (
        f"✅ Curated **{len(profiles)} speakers** from {int(n_samples)} clips of `{config}`. "
        "Pick one below to listen, then export."
    )
    choices = list(_last_profiles)
    return (
        _profiles_to_table(profiles),
        gr.update(choices=choices, value=choices[0] if choices else None),
        None,
        status,
    )


def curate_demo(min_score):
    """Offline mode: synthetic speakers, no HF account needed."""
    global _last_profiles
    from scripts.make_demo_data import rows

    profiles = build_profiles(rows(), min_score=min_score)
    _last_profiles = {f"{p.speaker_id} · {p.language} · score {p.best_score:.0f}": p for p in profiles}
    choices = list(_last_profiles)
    status = (
        f"✅ Demo mode: curated **{len(profiles)} synthetic speakers** "
        f"(each had 5 clips; low-quality ones were rejected by the scorer). "
        "This exercises the exact pipeline that runs on real Vaani audio."
    )
    return _profiles_to_table(profiles), gr.update(choices=choices, value=choices[0] if choices else None), None, status


def preview_voice(label):
    if not label or label not in _last_profiles:
        return None, ""
    p = _last_profiles[label]
    score, audio, sr, transcript = p.top_clips(1)[0]
    detail = (
        f"**{p.speaker_id}** — {p.language}, {p.gender or 'unspecified'}, "
        f"{p.district}, {p.state}\n\n"
        f"Best clip: {score.duration:.1f}s · {score.rms_dbfs:.0f} dBFS · "
        f"SNR {score.snr_db:.0f} dB · speech ratio {score.speech_ratio:.0%} · "
        f"**score {score.score:.0f}/100**"
        + (f"\n\n> {transcript}" if transcript.strip() else "")
    )
    return (sr, audio), detail


def export_selected(labels):
    if not labels:
        return "Select at least one voice to export."
    profiles = [_last_profiles[l] for l in labels if l in _last_profiles]
    created = export_voice_pack(profiles, PACKS_DIR)
    listing = "\n".join(f"- `{d.relative_to(PACKS_DIR.parent)}`" for d in created)
    return (
        f"📦 Exported **{len(created)} voice pack(s)**:\n{listing}\n\n"
        "Import into Voicebox: **Voices → Add Voice → choose `reference_1.wav`** "
        "as the cloning sample (Chatterbox Multilingual or Qwen3-TTS engines)."
    )


def clone_preview(label, text, lang_id):
    from vaanibox import tts

    if not tts.available():
        return None, (
            "Chatterbox not installed — cloning preview is optional. "
            "Install with `uv pip install chatterbox-tts` (~2 GB model on first run), "
            "or export the pack and clone inside Voicebox instead."
        )
    if not label or label not in _last_profiles:
        return None, "Curate and select a voice first."
    import tempfile

    import soundfile as sf

    p = _last_profiles[label]
    _, audio, sr, _ = p.top_clips(1)[0]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, sr)
        out_sr, out = tts.clone_speak(text, f.name, language_id=lang_id)
    return (out_sr, out), f"Generated with Chatterbox Multilingual, cloned from {p.speaker_id}."


with gr.Blocks(title="VaaniBox") as demo:
    gr.Markdown(
        "# 🇮🇳 VaaniBox\n"
        "**Every district of India, as a voice you can use.** Streams the "
        "[Vaani dataset](https://huggingface.co/datasets/ARTPARK-IISc/Vaani) "
        "(156K speakers, 165 districts, CC-BY-4.0), auto-curates clone-quality reference "
        "clips, and exports voice packs for "
        "[Voicebox](https://github.com/jamiepine/voicebox)."
        + ("" if has_hf_token() else
           "\n\n> ⚠️ No Hugging Face token detected — real Vaani streaming is disabled. "
           "Accept the dataset terms on Hugging Face and run `hf auth login`, or use **Demo mode** below.")
    )

    with gr.Tab("1 · Explore & curate"):
        with gr.Row():
            state = gr.Dropdown(STATES, label="State", value="Bihar")
            district = gr.Dropdown(districts_for("Bihar"), label="District", value="Araria")
            n_samples = gr.Slider(10, 200, value=40, step=10, label="Clips to scan")
            min_score = gr.Slider(0, 90, value=40, step=5, label="Min quality score")
        with gr.Row():
            curate_btn = gr.Button("🔍 Stream & curate from Vaani", variant="primary")
            demo_btn = gr.Button("🧪 Demo mode (offline, synthetic voices)")
        status = gr.Markdown()
        table = gr.Dataframe(
            headers=["Speaker", "Language", "Gender", "Clips kept", "Best score", "Duration", "SNR"],
            label="Curated speakers (best first)",
            interactive=False,
        )
        with gr.Row():
            with gr.Column():
                voice_pick = gr.Dropdown([], label="Listen to a curated voice")
                detail = gr.Markdown()
            player = gr.Audio(label="Best reference clip", interactive=False)

    with gr.Tab("2 · Export to Voicebox"):
        export_pick = gr.CheckboxGroup([], label="Voices to export")
        export_btn = gr.Button("📦 Export voice pack(s)", variant="primary")
        export_out = gr.Markdown()

    with gr.Tab("3 · Clone preview (optional)"):
        gr.Markdown(
            "Zero-shot cloning with **Chatterbox Multilingual** — the same engine Voicebox ships. "
            "Optional heavy dependency; the export path above needs nothing extra."
        )
        clone_text = gr.Textbox(label="Text to speak", value="नमस्ते, यह मेरी आवाज़ की एक झलक है।")
        clone_lang = gr.Dropdown(["hi", "ta", "te", "kn", "ml", "bn", "mr", "gu", "en"], value="hi", label="Language")
        clone_btn = gr.Button("🗣️ Generate in selected voice")
        clone_audio = gr.Audio(label="Cloned speech", interactive=False)
        clone_status = gr.Markdown()

    def _sync_districts(s):
        d = districts_for(s)
        return gr.update(choices=d, value=d[0] if d else None)

    def _sync_export(choices_update):
        return gr.update(choices=list(_last_profiles), value=[])

    state.change(_sync_districts, state, district)
    curate_btn.click(curate_district, [state, district, n_samples, min_score], [table, voice_pick, player, status]).then(
        _sync_export, voice_pick, export_pick
    )
    demo_btn.click(curate_demo, [min_score], [table, voice_pick, player, status]).then(
        _sync_export, voice_pick, export_pick
    )
    voice_pick.change(preview_voice, voice_pick, [player, detail])
    export_btn.click(export_selected, export_pick, export_out)
    clone_btn.click(clone_preview, [voice_pick, clone_text, clone_lang], [clone_audio, clone_status])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
