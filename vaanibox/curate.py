"""Clip quality scoring and speaker profiling.

Voice cloning quality depends almost entirely on the reference clip: it should
be a single speaker, 5-25 seconds, loud but not clipped, and mostly speech.
Vaani is spontaneous field-recorded speech, so a lot of it is too short, too
quiet, or too noisy to clone from. Everything here is plain numpy so it runs
on any machine with no model downloads.
"""

from dataclasses import dataclass, field

import numpy as np

FRAME_MS = 30

# Scoring targets for a good cloning reference clip.
IDEAL_DURATION = (5.0, 25.0)   # seconds; hard floor 3s, ceiling 40s
IDEAL_RMS_DBFS = (-30.0, -12.0)
IDEAL_SPEECH_RATIO = (0.40, 0.95)
MIN_SNR_DB = 10.0


@dataclass
class ClipScore:
    duration: float
    rms_dbfs: float
    clipping_ratio: float
    speech_ratio: float
    snr_db: float
    score: float  # 0-100 composite

    def as_dict(self) -> dict:
        return {k: round(v, 2) for k, v in self.__dict__.items()}


@dataclass
class VoiceProfile:
    """One Vaani speaker with their best reference clips."""

    speaker_id: str
    language: str = ""
    gender: str = ""
    state: str = ""
    district: str = ""
    clips: list = field(default_factory=list)  # (score: ClipScore, audio: np.ndarray, sr: int, transcript: str)

    @property
    def best_score(self) -> float:
        return max((c[0].score for c in self.clips), default=0.0)

    def top_clips(self, n: int = 3) -> list:
        return sorted(self.clips, key=lambda c: -c[0].score)[:n]


def _frame_energies(audio: np.ndarray, sr: int) -> np.ndarray:
    hop = max(1, int(sr * FRAME_MS / 1000))
    n_frames = len(audio) // hop
    if n_frames == 0:
        return np.zeros(1)
    frames = audio[: n_frames * hop].reshape(n_frames, hop)
    return np.sqrt(np.mean(frames**2, axis=1) + 1e-12)


def _band_score(value: float, lo: float, hi: float, falloff: float) -> float:
    """1.0 inside [lo, hi], linearly falling to 0 over `falloff` outside it."""
    if lo <= value <= hi:
        return 1.0
    dist = (lo - value) if value < lo else (value - hi)
    return float(max(0.0, 1.0 - dist / falloff))


def score_clip(audio: np.ndarray, sr: int) -> ClipScore:
    """Score a mono float clip on how well it would work as a cloning reference."""
    audio = np.asarray(audio, dtype=np.float32).flatten()
    peak = np.max(np.abs(audio)) if len(audio) else 0.0
    if peak > 1.0:  # some decoders return int-scaled floats
        audio = audio / peak

    duration = len(audio) / sr if sr else 0.0
    rms = float(np.sqrt(np.mean(audio**2) + 1e-12))
    rms_dbfs = 20 * np.log10(rms + 1e-12)
    clipping_ratio = float(np.mean(np.abs(audio) > 0.985)) if len(audio) else 1.0

    energies = _frame_energies(audio, sr)
    noise_floor = np.percentile(energies, 10)
    speech_level = np.percentile(energies, 90)
    snr_db = float(20 * np.log10((speech_level + 1e-9) / (noise_floor + 1e-9)))
    # Frames counted as speech if they are well above the noise floor.
    threshold = noise_floor * 3 + 1e-6
    speech_ratio = float(np.mean(energies > threshold))

    s_dur = _band_score(duration, *IDEAL_DURATION, falloff=8.0)
    s_loud = _band_score(rms_dbfs, *IDEAL_RMS_DBFS, falloff=12.0)
    s_speech = _band_score(speech_ratio, *IDEAL_SPEECH_RATIO, falloff=0.3)
    s_snr = min(1.0, max(0.0, (snr_db - 5.0) / (MIN_SNR_DB * 2 - 5.0)))
    s_clip = max(0.0, 1.0 - clipping_ratio * 200)

    score = 100 * (0.30 * s_dur + 0.20 * s_loud + 0.20 * s_speech + 0.20 * s_snr + 0.10 * s_clip)
    # A clip that fails any of these is unusable as a cloning reference no
    # matter how good it looks otherwise — gate, don't average.
    if duration < 3.0 or duration > 40.0:
        score *= 0.2
    if rms_dbfs < -40.0:
        score *= 0.5
    if snr_db < 5.0:
        score *= 0.6
    return ClipScore(duration, float(rms_dbfs), clipping_ratio, speech_ratio, snr_db, float(score))


def build_profiles(samples, min_score: float = 40.0, max_clips_per_speaker: int = 5) -> list:
    """Group scored samples into per-speaker VoiceProfiles.

    `samples` yields dicts with Vaani's columns: audio {array, sampling_rate},
    speakerID, language, gender, state, district, transcript.
    """
    profiles: dict[str, VoiceProfile] = {}
    for row in samples:
        audio = row["audio"]["array"]
        sr = row["audio"]["sampling_rate"]
        sc = score_clip(audio, sr)
        if sc.score < min_score:
            continue
        # Some Vaani rows have a null speakerID; fall back to the speaker image
        # hash so those clips still group into a stable profile.
        spk = str(row.get("speakerID") or row.get("speakerImageHash") or "unknown")[:24]
        prof = profiles.setdefault(
            spk,
            VoiceProfile(
                speaker_id=spk,
                language=str(row.get("language", "")),
                gender=str(row.get("gender", "")),
                state=str(row.get("state", "")),
                district=str(row.get("district", "")),
            ),
        )
        if len(prof.clips) < max_clips_per_speaker or sc.score > min(c[0].score for c in prof.clips):
            prof.clips.append((sc, np.asarray(audio, dtype=np.float32), sr, str(row.get("transcript", "") or "")))
            prof.clips = prof.top_clips(max_clips_per_speaker)
    return sorted(profiles.values(), key=lambda p: -p.best_score)
