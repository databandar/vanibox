"""Radio Vaani — AI radio and podcast in Indian district voices.

Run:  .venv/bin/python app.py   ->  http://127.0.0.1:5051

Content is pre-rendered by scripts/build_radio_demo.py (RJ + podcast in
Vaani district composite voices, plus everything in radio/music/).
"""

from pathlib import Path

from flask import Flask, render_template, send_from_directory

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)


@app.get("/")
@app.get("/radio")
def radio():
    return render_template("radio.html")


@app.get("/radio/playlist.json")
def radio_playlist():
    return send_from_directory(ROOT / "radio", "playlist.json")


@app.get("/radio/audio/<path:name>")
def radio_audio(name):
    return send_from_directory(ROOT / "radio", name)


if __name__ == "__main__":
    app.run(debug=True, port=5051)
