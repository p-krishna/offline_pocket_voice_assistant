"""
Simulated physical world for end-to-end testing.

  UserSim   — a "human" who speaks synthesized utterances (varied voices, speed,
              loudness, mid-sentence pauses) at chosen moments.
  Room      — mixes user speech + ambient noise + (optionally) the assistant's own
              loudspeaker output leaking back into the microphone (no AEC!).
  SimMic    — real-time paced microphone feeding the pipeline 30 ms frames.
  SimBackend— speaker backend that "plays" in real time into the Room and records
              exactly what a person standing there would have heard.

Everything runs in wall-clock real time on purpose: endpointing, cooldowns,
barge-in and progress narration are all timing behaviours.
"""
from __future__ import annotations

import hashlib
import io
import json
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from audio.mic import MicBase
from audio.speaker import SpeakerBackend

SR = 16000
CACHE_DIR = Path(__file__).resolve().parent / ".cache"


# ── user audio synthesis (cached; done BEFORE a scenario starts so the shared
#    TTS server is never contended while the assistant is working) ─────────────
def synth_user(text: str, voice: str = "am_adam", speed: float = 1.0,
               tts_url: str = "http://127.0.0.1:8082") -> np.ndarray:
    """Return float32 mono 16 kHz speech peak-normalised to 0.5."""
    CACHE_DIR.mkdir(exist_ok=True)
    key = hashlib.sha1(f"{text}|{voice}|{speed}".encode()).hexdigest()[:16]
    f = CACHE_DIR / f"{key}.npy"
    if f.exists():
        return np.load(f)
    req = urllib.request.Request(
        f"{tts_url}/synthesize",
        data=json.dumps({"text": text, "voice": voice, "speed": speed}).encode(),
        headers={"Content-Type": "application/json"})
    wav = urllib.request.urlopen(req, timeout=60).read()
    a, sr = sf.read(io.BytesIO(wav), dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != SR:
        a = resample_poly(a, SR, sr).astype(np.float32)
    peak = float(np.max(np.abs(a))) or 1.0
    a = (a / peak * 0.5).astype(np.float32)
    np.save(f, a)
    return a


@dataclass
class Clip:
    t_start: float            # monotonic
    pcm: np.ndarray           # float32 16 kHz
    label: str
    text: str = ""
    kind: str = "user"        # user | noise | bg

    @property
    def t_end(self) -> float:
        return self.t_start + len(self.pcm) / SR


class Room:
    def __init__(self, noise_rms: float = 0.002, echo_gain: float = 0.0, echo_delay_s: float = 0.06,
                 seed: int = 1):
        self.noise_rms = noise_rms
        self.echo_gain = echo_gain
        self.echo_delay_s = echo_delay_s
        self.rng = np.random.default_rng(seed)
        self.clips: List[Clip] = []
        self.all_clips: List[Clip] = []          # never pruned (ground truth)
        self.playback: List[Tuple[float, np.ndarray]] = []   # (t_start, float32 16k)
        self.lock = threading.Lock()
        self.mic_log: List[np.ndarray] = []
        self.t0 = time.monotonic()
        self._lp = np.zeros(2)

    def add_clip(self, clip: Clip) -> Clip:
        with self.lock:
            self.clips.append(clip)
            self.all_clips.append(clip)
        return clip

    def add_playback(self, t_start: float, pcm16: np.ndarray) -> None:
        with self.lock:
            self.playback.append((t_start, pcm16))

    def truncate_playback_after(self, t: float) -> None:
        """Speaker was stopped: nothing scheduled beyond `t` is audible any more."""
        with self.lock:
            out = []
            for ts, p in self.playback:
                if ts >= t:
                    continue
                n = int((t - ts) * SR)
                out.append((ts, p[:n]) if n < len(p) else (ts, p))
            self.playback = out

    def _mix(self, items, t0: float, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        t1 = t0 + n / SR
        for ts, p in items:
            te = ts + len(p) / SR
            if te <= t0 or ts >= t1:
                continue
            a = max(0, int(round((ts - t0) * SR)))
            b = min(n, int(round((te - t0) * SR)))
            pa = max(0, int(round((t0 - ts) * SR)))
            seg = p[pa: pa + (b - a)]
            out[a: a + len(seg)] += seg
        return out

    def render(self, t0: float, n: int) -> np.ndarray:
        with self.lock:
            clips = [(c.t_start, c.pcm) for c in self.clips if c.t_end > t0 - 1 and c.t_start < t0 + 1]
            pb = list(self.playback) if self.echo_gain > 0 else []
            # prune
            self.clips = [c for c in self.clips if c.t_end > t0 - 5]
            self.playback = [(ts, p) for ts, p in self.playback if ts + len(p) / SR > t0 - 5]
        sig = self._mix(clips, t0, n)
        if pb:
            sig += self.echo_gain * self._mix([(ts + self.echo_delay_s, p) for ts, p in pb], t0, n)
        if self.noise_rms > 0:
            w = self.rng.normal(0, 1, n).astype(np.float32)
            # crude low-pass → fan/room rumble character rather than pure hiss
            w = np.convolve(w, np.ones(6) / 6, mode="same")
            w *= self.noise_rms / (float(np.sqrt(np.mean(w ** 2))) + 1e-9)
            sig += w
        return np.clip(sig, -1.0, 1.0)


class SimMic(MicBase):
    sample_rate = SR

    def __init__(self, room: Room):
        self.room = room
        self._n = 0
        self._start = None
        self.stopped = False

    def open(self) -> None:
        self._start = time.monotonic()

    def read(self, n_samples: int) -> bytes:
        due = self._start + (self._n + n_samples) / SR
        while True:
            dt = due - time.monotonic()
            if dt <= 0 or self.stopped:
                break
            time.sleep(min(dt, 0.01))
        t0 = self._start + self._n / SR
        self._n += n_samples
        sig = self.room.render(t0, n_samples)
        self.room.mic_log.append(sig)
        return (sig * 32767).astype(np.int16).tobytes()

    def close(self) -> None:
        self.stopped = True


class SimBackend(SpeakerBackend):
    """A loudspeaker with a ~120 ms hardware buffer, played in real time."""
    BUFFER_S = 0.12
    native_sr = SR            # the Speaker resamples whole items to the room's 16 kHz with the same filter the AEC reference uses

    def __init__(self, room: Room):
        self.room = room
        self.record: List[Tuple[float, np.ndarray]] = []    # what a listener heard
        self._clock = 0.0       # time the device will be done with everything written
        self._lock = threading.Lock()

    def open(self, sr: int):
        return {"sr": sr}

    def write(self, handle, pcm_bytes: bytes) -> None:
        x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        now = time.monotonic()
        start = max(now, self._clock)
        self._clock = start + len(x) / SR
        self.room.add_playback(start, x)
        self.record.append((start, x))
        # Block like a real device: never run more than BUFFER_S ahead of real time.
        ahead = self._clock - now - self.BUFFER_S
        if ahead > 0:
            time.sleep(ahead)

    def close(self, handle, drain: bool) -> None:
        if not drain:
            now = time.monotonic()
            self.room.truncate_playback_after(now + 0.01)
            self.record = [(ts, p[: max(0, int((now + 0.01 - ts) * SR))]) for ts, p in self.record]
            self._clock = min(self._clock, now)
        else:
            rest = self._clock - time.monotonic()
            if rest > 0:
                time.sleep(rest)


# ── The simulated human ──────────────────────────────────────────────────────
class UserSim:
    def __init__(self, room: Room, default_voice: str = "am_adam"):
        self.room = room
        self.voice = default_voice
        self.said: List[dict] = []         # ground truth: what was spoken, and when

    def _schedule(self, pcm: np.ndarray, delay: float, label: str, text: str = "", kind="user", start_at=None) -> Clip:
        t = start_at if start_at is not None else time.monotonic() + delay
        c = self.room.add_clip(Clip(t, pcm, label, text, kind))
        if kind == "user":
            self.said.append({"text": text, "t_start": c.t_start, "t_end": c.t_end, "label": label})
        return c

    def say(self, text: str, voice: Optional[str] = None, speed: float = 1.0, gain: float = 0.6,
            delay: float = 0.0) -> Clip:
        a = synth_user(text, voice or self.voice, speed) * (gain / 0.5)
        return self._schedule(a, delay, label=text[:40], text=text)

    def say_parts(self, parts: List[Tuple[str, float]], voice: Optional[str] = None, speed: float = 1.0,
                  gain: float = 0.6, delay: float = 0.0) -> Clip:
        """Speak text fragments separated by pauses [(text, pause_after_s), ...] as ONE continuous
        performance (e.g. a person who hesitates mid-sentence)."""
        segs = []
        for text, pause in parts:
            segs.append(synth_user(text, voice or self.voice, speed) * (gain / 0.5))
            segs.append(np.zeros(int(pause * SR), dtype=np.float32))
        a = np.concatenate(segs)
        full = " ".join(t for t, _ in parts)
        return self._schedule(a, delay, label=full[:40], text=full)

    def noise_burst(self, dur_s: float = 0.4, level: float = 0.15, delay: float = 0.0, kind: str = "cough") -> Clip:
        n = int(dur_s * SR)
        w = self.room.rng.normal(0, 1, n).astype(np.float32)
        env = np.hanning(n).astype(np.float32)
        a = w * env * level
        return self._schedule(a, delay, label=kind, kind="noise")

    def background_speech(self, text: str, voice: str = "bf_emma", gain: float = 0.08, delay: float = 0.0) -> Clip:
        """TV / people talking elsewhere in the room."""
        a = synth_user(text, voice) * (gain / 0.5)
        return self._schedule(a, delay, label="bg:" + text[:25], text=text, kind="bg")
