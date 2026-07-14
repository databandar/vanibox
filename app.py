"""VaaniBox — Flask app. Explore Vaani's 165 districts, curate clone-ready
voices, export Voicebox voice packs.

Run:  .venv/bin/python app.py   ->  http://127.0.0.1:5051  (landing page; studio at /studio)
"""

import io
import re
import sys
from pathlib import Path

import soundfile as sf
from flask import Flask, abort, jsonify, render_template, request, send_file

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vaanibox import store
from vaanibox.curate import build_profiles
from vaanibox.dataset import VaaniAccessError, has_hf_token, stream_district
from vaanibox.districts import STATES, districts_for
from vaanibox.export import export_voice_pack

PACKS_DIR = Path(__file__).resolve().parent / "voice_packs"

app = Flask(__name__)

# The voice library: every curated profile, keyed by a URL-safe id.
# Backed by SQLite so it survives restarts.
_profiles: dict[str, object] = store.load_profiles()


def _pid(profile) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(profile.speaker_id))


def _profiles_json(profiles) -> list[dict]:
    out = []
    for p in profiles:
        best = p.top_clips(1)[0][0]
        out.append(
            {
                "id": _pid(p),
                "speaker": p.speaker_id,
                "language": p.language,
                "gender": p.gender,
                "district": p.district,
                "clips": len(p.clips),
                "score": round(p.best_score),
                "duration": round(best.duration, 1),
                "snr": round(best.snr_db),
            }
        )
    return out


def _store(profiles, persist: bool = True) -> None:
    new = {_pid(p): p for p in profiles}
    _profiles.update(new)  # merge into the library, don't wipe other districts
    if persist:
        store.save_profiles(new)


@app.get("/")
def landing():
    return render_template("landing.html")


@app.get("/studio")
def studio():
    return render_template(
        "index.html",
        states=STATES,
        districts=districts_for("Bihar"),
        hf_ok=has_hf_token(),
    )


@app.get("/api/districts/<state>")
def api_districts(state):
    return jsonify(districts_for(state))


@app.post("/api/curate")
def api_curate():
    body = request.get_json(force=True)
    config = f"{body['state']}_{body['district']}"
    limit = int(body.get("limit", 40))
    min_score = float(body.get("min_score", 40))
    try:
        profiles = build_profiles(stream_district(config, limit=limit), min_score=min_score)
    except VaaniAccessError as e:
        return jsonify({"error": str(e)}), 403
    except Exception as e:
        return jsonify({"error": f"Streaming failed: {e}"}), 502
    _store(profiles)
    return jsonify(
        {
            "speakers": _profiles_json(profiles),
            "status": f"Curated {len(profiles)} speakers from {limit} clips of {config}.",
        }
    )


@app.get("/api/library")
def api_library():
    profiles = sorted(_profiles.values(), key=lambda p: -p.best_score)
    return jsonify(
        {
            "speakers": _profiles_json(profiles),
            "status": f"Voice library: {len(profiles)} curated voice(s), from every district you've explored.",
        }
    )


@app.post("/api/curate-demo")
def api_curate_demo():
    from scripts.make_demo_data import rows

    min_score = float(request.get_json(force=True).get("min_score", 40))
    profiles = build_profiles(rows(), min_score=min_score)
    _store(profiles, persist=False)  # synthetic voices don't belong in the library
    return jsonify(
        {
            "speakers": _profiles_json(profiles),
            "status": f"Demo mode: curated {len(profiles)} synthetic speakers "
            "(low-quality clips rejected by the scorer).",
        }
    )


@app.get("/api/audio/<pid>")
def api_audio(pid):
    p = _profiles.get(pid)
    if p is None:
        abort(404)
    _, audio, sr, _ = p.top_clips(1)[0]
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    return send_file(buf, mimetype="audio/wav", download_name=f"{pid}.wav")


@app.post("/api/export")
def api_export():
    ids = request.get_json(force=True).get("ids", [])
    chosen = [_profiles[i] for i in ids if i in _profiles]
    if not chosen:
        return jsonify({"error": "Select at least one voice to export."}), 400
    created = export_voice_pack(chosen, PACKS_DIR)
    return jsonify(
        {
            "packs": [str(d.relative_to(PACKS_DIR.parent)) for d in created],
            "status": f"Exported {len(created)} voice pack(s). In Voicebox: "
            "Voices → Add Voice → choose reference_1.wav as the cloning sample.",
        }
    )


@app.post("/api/clone")
def api_clone():
    from vaanibox import tts

    if not tts.available():
        return jsonify(
            {"error": "Chatterbox not installed (optional). "
             "Run: uv pip install chatterbox-tts — or clone inside Voicebox."}
        ), 501
    from chatterbox.mtl_tts import SUPPORTED_LANGUAGES

    body = request.get_json(force=True)
    p = _profiles.get(body.get("id", ""))
    if p is None:
        return jsonify({"error": "Curate and select a voice first."}), 400
    lang = body.get("lang", "hi")
    if lang not in SUPPORTED_LANGUAGES:
        return jsonify(
            {"error": f"Chatterbox doesn't support '{lang}'. Indian languages: only Hindi "
             "(write in Devanagari) and English. For Marathi text, choose Hindi — the "
             "speaker's voice and accent still come from the reference clip."}
        ), 400
    text = (body.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Type some text to speak."}), 400
    import tempfile

    _, audio, sr, _ = p.top_clips(1)[0]
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            sf.write(f.name, audio, sr)
            out_sr, out = tts.clone_speak(text, f.name, lang)
    except Exception as e:
        return jsonify({"error": f"Generation failed: {e}"}), 500
    buf = io.BytesIO()
    sf.write(buf, out, out_sr, format="WAV")
    buf.seek(0)
    return send_file(buf, mimetype="audio/wav", download_name="cloned.wav")


if __name__ == "__main__":
    app.run(debug=True, port=5051)
