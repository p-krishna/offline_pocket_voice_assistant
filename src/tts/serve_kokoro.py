#!/usr/bin/env python3
"""
Kokoro TTS server — listens on port 8082.
POST /synthesize  with JSON body: { "text": "...", "voice": "af" }
Returns raw WAV bytes with Content-Type audio/wav.
"""

import sys
from pathlib import Path

# Allow running directly: add src/ to path so common.config is found.
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import io
import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro
from common.config import load_config


def _resolve_voices_path(path: str) -> str:
    """
    kokoro-onnx >= 0.5 loads voices with np.load() (an .npz/.bin archive),
    while this project ships voices.json. Convert the JSON once into a cached
    .npz next to it so both library generations work.
    """
    if not path.endswith(".json"):
        return path
    npz = Path(path).with_suffix(".npz")
    if not npz.exists() or npz.stat().st_mtime < Path(path).stat().st_mtime:
        print(f"[Kokoro] Converting {path} -> {npz} (one-time) ...")
        with open(path) as fh:
            raw = json.load(fh)
        np.savez(str(npz), **{k: np.array(v, dtype=np.float32) for k, v in raw.items()})
    return str(npz)


# Load config and model once at startup — reused for every request.
cfg = load_config()
print(f"[Kokoro] Loading model from {cfg.kokoro_model_path} ...")
tts = Kokoro(model_path=cfg.kokoro_model_path, voices_path=_resolve_voices_path(cfg.kokoro_voices_path))
print(f"[Kokoro] Model loaded. Listening on 127.0.0.1:{cfg.tts_server_port}")


def clean_text(text):
    # Strip markdown so spoken output sounds natural.
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)   # bold
    text = re.sub(r"\*(.+?)\*",       r"\1", text)   # italic
    text = re.sub(r"`(.+?)`",           r"\1", text)   # code
    text = re.sub(r"^#{1,6}\s+",       "",    text, flags=re.M)  # headings
    text = re.sub(r"^[-•*]\s+",        "",    text, flags=re.M)  # bullets
    text = re.sub(r"\s+",              " ",   text)   # collapse whitespace
    return text.strip()


def synthesize_to_wav_bytes(text, voice, speed=1.0):
    # Generate audio, then encode as WAV bytes in memory — no temp files.
    clean = clean_text(text)
    if not clean:
        return None
    audio, sample_rate = tts.create(clean, voice=voice, speed=speed)
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


class TTSHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Override to print cleaner logs.
        print(f"[Kokoro] {self.address_string()} - {fmt % args}")

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path != "/synthesize":
            self.send_response(404)
            self.end_headers()
            return

        try:
            # Read and parse JSON body.
            length = int(self.headers.get("Content-Length", 0))
            body   = json.loads(self.rfile.read(length).decode("utf-8"))
            text   = body.get("text", "").strip()
            voice  = body.get("voice", cfg.tts_voice)
            speed  = float(body.get("speed", 1.0))

            if not text:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing text")
                return

            wav_bytes = synthesize_to_wav_bytes(text, voice, speed)
            if wav_bytes is None:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Empty text after cleanup")
                return

            # Return raw WAV bytes.
            self.send_response(200)
            self.send_header("Content-Type",   "audio/wav")
            self.send_header("Content-Length", str(len(wav_bytes)))
            self.end_headers()
            self.wfile.write(wav_bytes)

        except Exception as e:
            print(f"[Kokoro] Error: {e}")
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode())


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", cfg.tts_server_port), TTSHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Kokoro] Stopped.")
