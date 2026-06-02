import io
import re
import struct
import time
import urllib.request
import json
from pathlib import Path
import math
import wave as wavemod

import numpy as np
import pyaudio
import soundfile as sf


class KokoroTTS:
    def __init__(self, cfg):
        self.url     = cfg.tts_server_url   # http://127.0.0.1:8082
        self.voice   = cfg.tts_voice
        self.out_dir = Path(cfg.tts_output_dir)
        self.timeout = cfg.http_timeout
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # this helper method — avoids duplicating the PyAudio logic in beep fallback:
    def _play_wav_bytes(self, wav_bytes: bytes) -> None:
        """Play raw WAV bytes via PyAudio. Shared by speak() and beep fallback."""
        buf = io.BytesIO(wav_bytes)
        audio_data, sample_rate = sf.read(buf, dtype="float32")
        pcm = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, output=True)
        stream.write(pcm)
        stream.stop_stream()
        stream.close()
        pa.terminate()

    # this method — stdlib only, no server needed:
    def _play_beep(self) -> None:
        """
        Play a short 440 Hz beep via PyAudio using stdlib only.
        Last-resort fallback when the TTS server is unreachable.
        """
        rate, duration_ms, freq = 22050, 350, 440
        n = int(rate * duration_ms / 1000)
        samples = [int(32767 * math.sin(2 * math.pi * freq * i / rate)) for i in range(n)]
        buf = io.BytesIO()
        with wavemod.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(struct.pack(f"<{n}h", *samples))
        buf.seek(0)
        self._play_wav_bytes(buf.read())

    def _synthesize(self, text: str) -> bytes | None:
        """
        Send one text chunk to Kokoro server, return WAV bytes.
        Returns None and plays a beep on failure.
        """
        payload = json.dumps({"text": text, "voice": self.voice}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.url}/synthesize",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read()
        except Exception as e:
            print(f"[TTS] Request failed: {e}")
            try:
                self._play_beep()
            except Exception as beep_err:
                print(f"[TTS] Beep fallback also failed: {beep_err}")
            return None

    def speak(self, text: str) -> None:
        """Synthesize full text in one request and play it. Used for short fallback phrases."""
        if not text:
            return
        wav_bytes = self._synthesize(text)
        if wav_bytes is None:
            return

        # Save WAV to disk.
        path = self.out_dir / f"tts_{int(time.time())}.wav"
        path.write_bytes(wav_bytes)
        print(f"[TTS] Saved: {path}")

        # Play via the shared helper.
        self._play_wav_bytes(wav_bytes)
        print(f"[TTS] Played: {text[:60]}...")

    def speak_streaming(self, text: str) -> None:
        """
        Split text into sentences and synthesize + play each one immediately.
        This cuts perceived latency — audio starts before the full response is ready.
        Each sentence is a separate HTTP request to the Kokoro server.

        Used for normal LLM responses. speak() is still used for short fallback phrases
        where splitting would produce awkward pauses.
        """
        if not text:
            return

        # Split on sentence boundaries: . ! ? followed by space or end of string.
        # Keeps the punctuation attached to the sentence before the split.
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

        # If no sentence boundaries found (e.g. a single short answer), speak whole text.
        if not sentences:
            self.speak(text)
            return

        for sentence in sentences:
            wav_bytes = self._synthesize(sentence)
            if wav_bytes is None:
                # One sentence failed — stop here, beep already played inside _synthesize.
                break
            self._play_wav_bytes(wav_bytes)
            print(f"[TTS] Streamed: {sentence[:60]}...")


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
    tts.speak_streaming(
        "Hello! This is a streaming test of the Kokoro TTS server. "
        "Each sentence should start playing immediately. "
        "If you hear all three sentences, streaming is working correctly."
    )