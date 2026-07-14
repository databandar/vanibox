"""Optional zero-shot cloning preview.

Uses Chatterbox Multilingual (same engine Voicebox ships) if installed:
    uv pip install chatterbox-tts
The model is ~2 GB on first run, so it is strictly opt-in; the rest of
VaaniBox works without it.
"""

_model = None


def available() -> bool:
    try:
        import chatterbox  # noqa: F401

        return True
    except ImportError:
        return False


def clone_speak(text: str, reference_wav: str, language_id: str = "hi") -> tuple[int, "np.ndarray"]:
    """Generate `text` in the voice of `reference_wav`. Returns (sr, audio)."""
    global _model
    import torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    if _model is None:
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        _model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    wav = _model.generate(text, audio_prompt_path=reference_wav, language_id=language_id)
    return _model.sr, wav.squeeze(0).cpu().numpy()
