"""
System phrase cache.

Short status phrases ("Let me think about that", "Sorry, I didn't catch that")
are synthesized once and stored as WAV files on disk, so they play instantly
with no TTS round-trip — which matters most exactly when something is slow or
broken (the TTS server might be the thing that is down).

Cache files are named <key>.<hash-of-text>.wav, so editing a phrase's text
automatically invalidates its cached audio.

Usage:
    phrases = PhrasePlayer(cfg)
    phrases.warm_up()              # once, after wait_for_servers()
    phrases.get("thinking")        # → (int16 pcm, sample_rate) or None
    phrases.pick("progress")       # → (key, pcm, sr), rotates through variants
"""

import hashlib
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

from common.log import get_logger

log = get_logger("phrases")

# Canonical phrase keys → text to synthesize.
PHRASES = {
    # --- session ---
    "yes":             "Yes?",
    "listening":       "I'm listening.",
    "going_to_sleep":  "Are you still there? I'll go to sleep soon.",
    "goodbye":         "Okay, going to sleep. Just say the wake word when you need me.",
    "okay":            "Okay.",
    # --- slow inference narration ---
    "thinking":        "Let me think about that.",
    "thinking_2":      "One moment please.",
    "still_working":   "Still working on it.",
    "still_working_2": "This is taking a little longer than usual. Thanks for waiting.",
    "almost":          "Almost there.",
    "long_wait":       "Sorry, this is a hard one. Still thinking.",
    # --- recoverable problems ---
    "didnt_catch":     "Sorry, I didn't catch that. Could you say it again?",
    "i_heard_you":     "I heard you, one moment",
    "repeat_that":     "Could you repeat that",
    "please_wait":     "Please wait, still thinking",
    "nothing_to_repeat": "I haven't said anything yet.",
    # --- failures (always spoken, so the assistant never goes silently dead) ---
    "fallback_stt":     "Sorry, my hearing is having trouble. Please try again in a moment.",
    "fallback_llm":     "Sorry, I'm having trouble thinking right now. Please try again in a moment.",
    "fallback_tts":     "Sorry, my voice is having trouble. Please try again.",
    "fallback_generic": "Sorry, something went wrong. Please try again.",
    "llm_timeout":      "Sorry, that took too long and I had to give up. Could you ask again, maybe more simply?",
}

# Groups used to rotate wording so repeated waits don't sound robotic.
GROUPS = {
    "progress_first": ["thinking", "thinking_2"],
    "progress_next":  ["still_working", "still_working_2"],
    "progress_long":  ["long_wait", "almost"],
}


def _fname(key: str, text: str) -> str:
    return f"{key}.{hashlib.sha1(text.encode()).hexdigest()[:8]}.wav"


class PhrasePlayer:
    def __init__(self, cfg):
        self.url     = cfg.tts_server_url
        self.voice   = cfg.system_voice
        self.timeout = cfg.http_timeout

        # Cache dir: <tts_output_dir>/phrases/<voice>/ — keyed by voice so
        # switching voices forces fresh synthesis.
        self.cache_dir = Path(cfg.tts_output_dir) / "phrases" / self.voice
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._cache: dict[str, tuple[np.ndarray, int]] = {}
        self._rot: dict[str, int] = {}

    # ── Startup ──────────────────────────────────────────────────────────────

    def warm_up(self) -> None:
        """Load every phrase from disk, synthesizing (once) any that are missing."""
        from audio.speaker import pcm_from_wav_bytes
        t0 = time.monotonic()
        synthesized = 0
        for key, text in PHRASES.items():
            path = self.cache_dir / _fname(key, text)
            wav = None
            if path.exists():
                wav = path.read_bytes()
            else:
                wav = self._synthesize(text)
                if wav:
                    path.write_bytes(wav)
                    synthesized += 1
                else:
                    log.warning("could not synthesize phrase '%s' — it will be silent", key)
            if wav:
                try:
                    self._cache[key] = pcm_from_wav_bytes(wav)
                except Exception as e:
                    log.warning("bad cached phrase '%s': %s", key, e)
        log.info("phrases ready: %d/%d (%d newly synthesized) in %.1fs",
                 len(self._cache), len(PHRASES), synthesized, time.monotonic() - t0)

    # ── Lookup ───────────────────────────────────────────────────────────────

    def get(self, key: str):
        """(int16 pcm, sample_rate) for a phrase, or None if unavailable."""
        return self._cache.get(key)

    def pick(self, group: str):
        """Next variant of a phrase group → (key, pcm, sr) or None."""
        keys = [k for k in GROUPS.get(group, [group]) if k in self._cache]
        if not keys:
            return None
        i = self._rot.get(group, 0)
        self._rot[group] = i + 1
        k = keys[i % len(keys)]
        pcm, sr = self._cache[k]
        return k, pcm, sr

    # ── Standalone playback (used only by this module's self-test) ───────────

    def play(self, key: str) -> None:
        import pyaudio
        item = self.get(key)
        if not item:
            print(f"[phrases] No cache for '{key}', skipping")
            return
        pcm, sr = item
        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=sr, output=True, frames_per_buffer=4096)
        stream.write(pcm.tobytes())
        stream.stop_stream()
        stream.close()
        pa.terminate()

    # ── Internals ────────────────────────────────────────────────────────────

    def _synthesize(self, text: str) -> bytes | None:
        payload = json.dumps({"text": text, "voice": self.voice}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.url}/synthesize",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=max(self.timeout, 30)) as resp:
                return resp.read()
        except Exception as e:
            log.warning("synthesis failed for '%s': %s", text, e)
            return None


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
