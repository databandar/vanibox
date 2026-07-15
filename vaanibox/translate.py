"""'My voice, another language' — the magic trick.

A voice note in Hindi/Marathi/anything becomes the same person speaking
English: Whisper transcribes AND translates in one pass (its built-in
translate task), then Chatterbox TTS speaks the English text using the
original voice note as the cloning reference. Everything runs locally.
"""

_whisper = None


def _model():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel

        # small = good multilingual accuracy at ~2-4s per voice note on CPU
        _whisper = WhisperModel("small", device="cpu", compute_type="int8")
    return _whisper


def to_english_text(wav_path: str) -> tuple[str, str]:
    """Transcribe + translate any-language speech. Returns (english, lang)."""
    segments, info = _model().transcribe(wav_path, task="translate", beam_size=3, vad_filter=True)
    text = " ".join(s.text.strip() for s in segments).strip()
    return text, info.language


def speak_translated(wav_path: str):
    """Voice note in -> (sr, audio, english_text, detected_lang): the same
    voice speaking English."""
    from vaanibox import tts

    english, lang = to_english_text(wav_path)
    if not english:
        raise ValueError("Couldn't hear any speech in that clip.")
    sr, audio = tts.clone_speak(english, wav_path, language_id="en")
    return sr, audio, english, lang
