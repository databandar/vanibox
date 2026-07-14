"""SQLite persistence for curated voices.

Curation costs streaming time, so keep every curated voice in a local library
(vaanibox.db) that survives server restarts. Audio is stored wav-encoded; a
voice's clips are replaced wholesale on re-curation of the same speaker.
"""

import io
import json
import sqlite3
from pathlib import Path

import numpy as np
import soundfile as sf

from .curate import ClipScore, VoiceProfile

DB_PATH = Path(__file__).resolve().parents[1] / "vaanibox.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS voices (
    id TEXT PRIMARY KEY,
    speaker_id TEXT NOT NULL,
    language TEXT, gender TEXT, state TEXT, district TEXT,
    best_score REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS clips (
    voice_id TEXT NOT NULL REFERENCES voices(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    score_json TEXT NOT NULL,
    transcript TEXT,
    sample_rate INTEGER NOT NULL,
    wav BLOB NOT NULL,
    PRIMARY KEY (voice_id, idx)
);
"""


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(_SCHEMA)
    return con


def save_profiles(profiles: dict) -> None:
    """Upsert {pid: VoiceProfile} into the library."""
    with _connect() as con:
        for pid, p in profiles.items():
            con.execute(
                "INSERT INTO voices (id, speaker_id, language, gender, state, district, best_score) "
                "VALUES (?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                "language=excluded.language, gender=excluded.gender, state=excluded.state, "
                "district=excluded.district, best_score=excluded.best_score",
                (pid, p.speaker_id, p.language, p.gender, p.state, p.district, p.best_score),
            )
            con.execute("DELETE FROM clips WHERE voice_id = ?", (pid,))
            for i, (score, audio, sr, transcript) in enumerate(p.top_clips(5)):
                buf = io.BytesIO()
                sf.write(buf, np.asarray(audio, dtype=np.float32), sr, format="WAV")
                con.execute(
                    "INSERT INTO clips (voice_id, idx, score_json, transcript, sample_rate, wav) "
                    "VALUES (?,?,?,?,?,?)",
                    (pid, i, json.dumps(score.as_dict()), transcript, sr, buf.getvalue()),
                )


def load_profiles() -> dict:
    """Return the whole library as {pid: VoiceProfile}, best voices first."""
    if not DB_PATH.exists():
        return {}
    out: dict[str, VoiceProfile] = {}
    with _connect() as con:
        rows = con.execute(
            "SELECT id, speaker_id, language, gender, state, district FROM voices "
            "ORDER BY best_score DESC"
        ).fetchall()
        for pid, speaker_id, language, gender, state, district in rows:
            prof = VoiceProfile(
                speaker_id=speaker_id, language=language or "", gender=gender or "",
                state=state or "", district=district or "",
            )
            for score_json, transcript, sr, wav in con.execute(
                "SELECT score_json, transcript, sample_rate, wav FROM clips "
                "WHERE voice_id = ? ORDER BY idx", (pid,)
            ):
                audio, _ = sf.read(io.BytesIO(wav), dtype="float32")
                prof.clips.append((ClipScore(**json.loads(score_json)), audio, sr, transcript or ""))
            if prof.clips:
                out[pid] = prof
    return out
