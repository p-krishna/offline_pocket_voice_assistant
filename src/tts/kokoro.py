import io
import time
import urllib.request
import json
from pathlib import Path

import numpy as np
import pyaudio
import soundfile as sf


class KokoroTTS:
    def __init__(self, cfg):
        self.url     = cfg.tts_server_url   # http://127.0.0.1:8082
        self.voice   = cfg.tts_voice
        self.out_dir = Path(cfg.tts_output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def speak(self, text):
        if not text:
            return

        # Send text to Kokoro TTS server and receive raw WAV bytes.
        payload = json.dumps({"text": text, "voice": self.voice}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.url}/synthesize",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                wav_bytes = resp.read()
        except Exception as e:
            print(f"[TTS] Request failed: {e}")
            return

        # Save WAV to disk.
        path = self.out_dir / f"tts_{int(time.time())}.wav"
        path.write_bytes(wav_bytes)
        print(f"[TTS] Saved: {path}")

        # Play WAV immediately via PyAudio.
        buf = io.BytesIO(wav_bytes)
        audio_data, sample_rate = sf.read(buf, dtype="float32")
        pcm = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

        pa     = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, output=True)
        stream.write(pcm)
        stream.stop_stream()
        stream.close()
        pa.terminate()
        print(f"[TTS] Played: {text[:60]}...")


# --- Standalone test ---
if __name__ == "__main__":
    import sys
    from pathlib import Path
    _SRC = Path(__file__).resolve().parent.parent
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    from common.config import load_config

    cfg = load_config()
    tts = KokoroTTS(cfg)
    tts.speak(
        "Hello! This is a test of the Kokoro TTS server. "
        "If you can hear this, the text to speech server is working correctly."
    )
