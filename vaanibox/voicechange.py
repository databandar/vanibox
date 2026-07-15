"""Voice conversion for the messaging bot.

ChatterboxVC re-voices speech directly (no transcription step, so it works in
any language and keeps the original pacing). The default characters build on
the engine's bundled studio voice — the same default voice Voicebox ships —
plus classic post effects. Any .wav dropped into characters/ becomes an extra
character, used as the VC target voice.
"""

from pathlib import Path

import numpy as np

CHARACTERS_DIR = Path(__file__).resolve().parents[1] / "characters"

# name -> (emoji, description, effect). Effects run on the VC output.
DEFAULT_CHARACTERS = {
    "anchor": ("🎙", "Studio anchor — the engine's default voice", None),
    "deep": ("🐻", "Deep — anchor pitched down", ("pitch", -4)),
    "chipmunk": ("🐿", "Chipmunk — anchor pitched up", ("pitch", +6)),
    "robot": ("🤖", "Robot — anchor with a metallic ring", ("robot", None)),
}

_vc = None


def _model():
    global _vc
    if _vc is None:
        import torch
        from chatterbox.vc import ChatterboxVC

        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        _vc = ChatterboxVC.from_pretrained(device=device)
    return _vc


def list_characters() -> dict[str, str]:
    """{key: label} — defaults plus any custom wav in characters/."""
    out = {k: f"{e} {k.title()}" for k, (e, _, _) in DEFAULT_CHARACTERS.items()}
    if CHARACTERS_DIR.is_dir():
        for wav in sorted(CHARACTERS_DIR.glob("*.wav")):
            out[f"custom:{wav.stem}"] = f"🗣 {wav.stem.replace('_', ' ').title()}"
    return out


def _apply_effect(audio: np.ndarray, sr: int, effect) -> np.ndarray:
    if effect is None:
        return audio
    kind, arg = effect
    if kind == "pitch":
        import librosa

        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=arg)
    if kind == "robot":
        t = np.arange(len(audio)) / sr
        ring = np.sin(2 * np.pi * 70 * t).astype(np.float32)
        mixed = audio * (0.55 + 0.45 * ring)
        return mixed / (np.max(np.abs(mixed)) + 1e-6) * np.max(np.abs(audio))
    return audio


def convert(input_wav: str, character: str) -> tuple[int, np.ndarray]:
    """Re-voice `input_wav` as `character`. Returns (sr, mono float32)."""
    model = _model()
    if character.startswith("custom:"):
        target = CHARACTERS_DIR / f"{character.split(':', 1)[1]}.wav"
        if not target.exists():
            raise ValueError(f"Unknown character {character}")
        wav = model.generate(input_wav, target_voice_path=str(target))
        effect = None
    else:
        if character not in DEFAULT_CHARACTERS:
            raise ValueError(f"Unknown character {character}")
        wav = model.generate(input_wav)  # bundled default voice as target
        effect = DEFAULT_CHARACTERS[character][2]
    audio = wav.squeeze(0).cpu().numpy().astype(np.float32)
    return model.sr, _apply_effect(audio, model.sr, effect)
