"""VaaniBox voice-changer bot for Telegram.

Send the bot a voice note; it re-voices it as your chosen character and sends
it back as a voice note. Voice conversion runs locally (ChatterboxVC); nothing
leaves your machine except the Telegram messages themselves.

Setup:
  1. Create a bot with @BotFather in Telegram, copy the token.
  2. Put it in .telegram_token (gitignored) or export TELEGRAM_BOT_TOKEN.
  3. Run:  .venv/bin/python tg_bot.py

Characters: four defaults built on the engine's bundled studio voice
(anchor / deep / chipmunk / robot), plus any .wav you drop into characters/.
"""

import asyncio
import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import soundfile as sf
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from vaanibox.voicechange import convert, list_characters

TOKEN_FILE = ROOT / ".telegram_token"

# chat_id -> character key; default is the plain studio anchor
_chosen: dict[int, str] = {}


def _token() -> str:
    tok = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not tok and TOKEN_FILE.exists():
        tok = TOKEN_FILE.read_text().strip()
    if not tok:
        raise SystemExit(
            "No Telegram token. Create a bot with @BotFather, then either\n"
            "  echo '<token>' > .telegram_token\n"
            "or export TELEGRAM_BOT_TOKEN before running."
        )
    return tok


def _keyboard() -> InlineKeyboardMarkup:
    rows, row = [], []
    for key, label in list_characters().items():
        row.append(InlineKeyboardButton(label, callback_data=f"char:{key}"))
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    return InlineKeyboardMarkup(rows)


async def cmd_start(update: Update, _: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🎙 VaaniBox voice changer\n\n"
        "Pick a character, then send me a voice note — I'll send it back "
        "spoken in that voice. Conversion runs locally on the host machine.",
        reply_markup=_keyboard(),
    )


async def cmd_characters(update: Update, _: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Pick a character:", reply_markup=_keyboard())


async def on_pick(update: Update, _: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    key = q.data.split(":", 1)[1]
    _chosen[q.message.chat_id] = key
    await q.answer()
    await q.edit_message_text(
        f"Character set to {list_characters().get(key, key)}. Now send me a voice note!"
    )


def _to_opus(wav_bytes: bytes) -> bytes | None:
    """Encode wav -> OGG/Opus so Telegram renders a proper voice note."""
    try:
        p = subprocess.run(
            ["ffmpeg", "-i", "pipe:0", "-c:a", "libopus", "-b:a", "40k", "-f", "ogg", "pipe:1"],
            input=wav_bytes, capture_output=True, timeout=60, check=True,
        )
        return p.stdout
    except Exception:
        return None


def _convert_blocking(ogg_bytes: bytes, character: str) -> bytes:
    """voice note bytes -> converted wav bytes (runs in a worker thread)."""
    audio, sr = sf.read(io.BytesIO(ogg_bytes), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        sf.write(f.name, audio, sr)
        out_sr, out = convert(f.name, character)
    buf = io.BytesIO()
    sf.write(buf, out, out_sr, format="WAV")
    return buf.getvalue()


async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    character = _chosen.get(msg.chat_id, "anchor")
    voice = msg.voice or msg.audio
    if voice.duration and voice.duration > 60:
        await msg.reply_text("Keep it under 60 seconds, please.")
        return
    note = await msg.reply_text(
        f"🎛 Re-voicing as {list_characters().get(character, character)}… (~20–40s)"
    )
    try:
        tg_file = await context.bot.get_file(voice.file_id)
        ogg = bytes(await tg_file.download_as_bytearray())
        wav = await asyncio.to_thread(_convert_blocking, ogg, character)
        opus = await asyncio.to_thread(_to_opus, wav)
        if opus:
            await msg.reply_voice(opus, caption=list_characters().get(character, character))
        else:  # no ffmpeg — send as a playable audio file instead
            await msg.reply_audio(wav, filename="voice.wav",
                                  title=list_characters().get(character, character))
        await note.delete()
    except Exception as e:
        await note.edit_text(f"⚠️ Conversion failed: {e}")


async def on_text(update: Update, _: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Send me a voice note 🎤 (or /characters to switch voices)."
    )


def main():
    app = (
        Application.builder()
        .token(_token())
        # generous timeouts + retries: a brief network stall at launch should
        # mean "try again", not a crash (default is one attempt, 5s timeouts)
        .connect_timeout(20)
        .read_timeout(30)
        .write_timeout(30)
        .get_updates_connect_timeout(20)
        .get_updates_read_timeout(60)
        .build()
    )
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("characters", cmd_characters))
    app.add_handler(CallbackQueryHandler(on_pick, pattern=r"^char:"))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, on_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    print("VaaniBox voice-changer bot running — send /start to your bot in Telegram.")
    app.run_polling(allowed_updates=Update.ALL_TYPES, bootstrap_retries=-1)


if __name__ == "__main__":
    main()
