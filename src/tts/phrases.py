"""
System phrase cache.

Synthesizes short status phrases once at startup and stores them as WAV
files on disk. On subsequent runs the files are reused — no HTTP call needed.

All phrases use cfg.system_voice so switching voices later requires only a
config change (and deleting the cache dir to force re-synthesis).

Usage:
    from tts.phrases import PhrasePlayer
    phrases = PhrasePlayer(cfg)
    phrases.warm_up()          # call once after wait_for_servers()
    phrases.play("listening")  # non-blocking, plays in caller's thread
"""

import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pyaudio
import soundfile as sf


# Canonical phrase keys → text to synthesize.
PHRASES = {
    "listening":       "Listening",
    "thinking":        "Thinking",
    "i_heard_you":     "I heard you, one moment",
    "repeat_that":     "Could you repeat that",
    "please_wait":     "Please wait, still thinking",
    "going_to_sleep":  "Going to sleep soon",
    "goodbye":         "I am going to sleep, goodbye",
}


class PhrasePlayer:
    def __init__(self, cfg):
        self.url     = cfg.tts_server_url
        self.voice   = cfg.system_voice
        self.timeout = cfg.http_timeout

        # Cache dir: debug_audio/phrases/<voice>/
        # Keyed by voice so switching voices forces fresh synthesis.
        self.cache_dir = Path(cfg.tts_output_dir) / "phrases" / self.voice
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory WAV bytes, populated by warm_up().
        self._cache: dict[str, bytes] = {}

    # ── Startup ──────────────────────────────────────────────────────────────

    def warm_up(self) -> None:
        """
        For each phrase:
          - If the WAV file already exists on disk, load it.
          - Otherwise synthesize via Kokoro server and save to disk.
        Call this once after wait_for_servers() returns.
        """
        for key, text in PHRASES.items():
            path = self.cache_dir / f"{key}.wav"
            if path.exists():
                # Reuse existing file — no HTTP call needed.
                self._cache[key] = path.read_bytes()
                print(f"[phrases] Loaded from cache: {key}")
            else:
                # Synthesize and persist.
                wav = self._synthesize(text)
                if wav:
                    path.write_bytes(wav)
                    self._cache[key] = wav
                    print(f"[phrases] Synthesized and cached: {key}")
                else:
                    print(f"[phrases] WARNING: could not synthesize '{key}' — will be silent")

    # ── Playback ─────────────────────────────────────────────────────────────

    def play(self, key: str) -> None:
        """
        Play a cached phrase by key.
        Blocks until audio finishes — call from a thread if needed.
        Silently skips if the key was not cached (e.g. synthesis failed).
        """
        wav = self._cache.get(key)
        if not wav:
            print(f"[phrases] No cache for '{key}', skipping")
            return
        self._play_wav_bytes(wav)

    # ── Internals ────────────────────────────────────────────────────────────

    def _synthesize(self, text: str) -> bytes | None:
        """Send text to Kokoro server, return WAV bytes or None on failure."""
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
            print(f"[phrases] Synthesis failed for '{text}': {e}")
            return None

    def _play_wav_bytes(self, wav_bytes: bytes) -> None:
        """Play raw WAV bytes via PyAudio with a buffer size that avoids underruns."""
        import io
        buf = io.BytesIO(wav_bytes)
        audio_data, sample_rate = sf.read(buf, dtype="float32")
        pcm = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            output=True,
            frames_per_buffer=4096,
        )
        stream.write(pcm)
        stream.stop_stream()
        stream.close()
        pa.terminate()


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    _SRC = Path(__file__).resolve().parent.parent
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    from common.config import load_config
    from common.servers import wait_for_servers

    cfg = load_config()
    wait_for_servers()

    player = PhrasePlayer(cfg)
    player.warm_up()

    for key in PHRASES:
        print(f"Playing: {key}")
        player.play(key)
        time.sleep(0.4)
