"""
Single serialised audio output.

Every sound the assistant makes (earcons, cached phrases, TTS sentences) goes
through ONE queue and ONE worker thread, so sounds can never overlap or race
each other, and one `stop()` call silences everything instantly (barge-in).

Each queued item carries the playback `epoch` it was created in. `stop()` bumps
the epoch, so anything still in flight from the cancelled turn (a TTS sentence
that finishes synthesising late, a filler phrase) is dropped on arrival.
"""
from __future__ import annotations

import collections
import io
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from common.log import get_logger, trace

log = get_logger("speaker")

CHUNK_S = 0.020   # write granularity: bounds stop latency to ~one chunk + device buffer


@dataclass
class AudioItem:
    kind: str                      # "earcon" | "phrase" | "tts"
    pcm: np.ndarray                # int16 mono
    sr: int
    label: str = ""
    epoch: Optional[int] = None    # None → current epoch at enqueue time
    on_start: Optional[Callable[["AudioItem"], None]] = None
    on_end: Optional[Callable[["AudioItem"], None]] = None
    # filled in by the speaker
    completed: bool = False
    started_at: float = 0.0
    ended_at: float = 0.0
    done: threading.Event = field(default_factory=threading.Event)

    @property
    def duration_s(self) -> float:
        return len(self.pcm) / float(self.sr)


class SpeakerBackend:
    """Output device. Implementations: PyAudioBackend (real), SimBackend (tests)."""

    # If set, the Speaker resamples every item ONCE (high quality) to this rate before writing, instead
    # of the backend converting chunk by chunk. Real devices accept any rate and leave this None.
    native_sr: Optional[int] = None

    def open(self, sr: int): raise NotImplementedError
    def write(self, handle, pcm_bytes: bytes) -> None: raise NotImplementedError
    def close(self, handle, drain: bool) -> None: raise NotImplementedError


class PyAudioBackend(SpeakerBackend):
    """
    Real loudspeaker. One output stream per sample rate is kept open between items so
    consecutive sentences play gaplessly (re-opening a device per sentence costs
    50–300 ms of silence and can click). Aborted playback flushes the device buffer.
    """

    IDLE_CLOSE_S = 30.0

    def __init__(self):
        import pyaudio
        self._pyaudio = pyaudio
        self._pa = pyaudio.PyAudio()
        self._stream = None
        self._sr = None
        self._last_used = 0.0
        self._lock = threading.Lock()

    def open(self, sr: int):
        with self._lock:
            now = time.monotonic()
            if self._stream is not None and (self._sr != sr or now - self._last_used > self.IDLE_CLOSE_S):
                self._close_locked()
            if self._stream is None:
                self._stream = self._pa.open(format=self._pyaudio.paInt16, channels=1, rate=sr,
                                             output=True, frames_per_buffer=1024)
                self._sr = sr
            self._last_used = now
            return self._stream

    def write(self, handle, pcm_bytes: bytes) -> None:
        handle.write(pcm_bytes)
        self._last_used = time.monotonic()

    def close(self, handle, drain: bool) -> None:
        with self._lock:
            try:
                if not drain and self._stream is not None:
                    self._stream.stop_stream()      # drop whatever is still buffered
                    self._stream.start_stream()
            except Exception as e:
                log.debug("stream flush: %s", e)
                self._close_locked()

    def _close_locked(self) -> None:
        try:
            if self._stream is not None:
                self._stream.close()
        except Exception as e:
            log.debug("stream close: %s", e)
        self._stream = None
        self._sr = None


def pcm_from_wav_bytes(wav: bytes):
    """Decode WAV bytes → (int16 mono ndarray, sample_rate)."""
    import soundfile as sf
    data, sr = sf.read(io.BytesIO(wav), dtype="float32", always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    return (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16), int(sr)


class Speaker:
    def __init__(self, backend: SpeakerBackend, drain_extra_s: float = 0.15, pause_timeout_s: float = 40.0):
        self.backend = backend
        self.drain_extra_s = drain_extra_s
        # A pause that nobody resolves (a crashed confirmation thread, a dead STT server) must not leave the
        # assistant mute for good: after this long the answer simply continues.
        self.pause_timeout_s = pause_timeout_s
        self._paused = False
        self._paused_at = 0.0
        self._q: "queue.Queue[AudioItem]" = queue.Queue()
        self._lock = threading.Lock()
        self.epoch = 0
        self._abort = threading.Event()
        self._current: Optional[AudioItem] = None
        self._levels = collections.deque(maxlen=64)   # (monotonic, rms)
        self.last_end = 0.0
        # Observers of everything sent to the loudspeaker: fn(item_started_at, sample_offset, int16_chunk, sr, item, (gain0, gain1)).
        # The echo canceller uses this as its far-end reference.
        self.taps: list = []
        self._duck = 1.0            # target gain (barge-in ducking)
        self._gain = 1.0            # gain applied so far (ramped, no clicks)
        self._running = True
        self._idle = threading.Event()
        self._idle.set()
        self._pending = 0
        self._thread = threading.Thread(target=self._run, name="SpeakerThread", daemon=True)
        self._thread.start()

    # ── public API ────────────────────────────────────────────────────────
    def enqueue(self, item: AudioItem) -> AudioItem:
        with self._lock:
            if item.epoch is None:
                item.epoch = self.epoch
            if item.epoch < self.epoch:
                log.debug("drop stale %s '%s' (epoch %d < %d)", item.kind, item.label, item.epoch, self.epoch)
                self._finish_dropped(item)
                return item
            self._pending += 1
            self._idle.clear()
        self._q.put(item)
        return item

    def pause(self, reason: str = "") -> None:
        """
        Fade out and hold the current answer exactly where it is (nothing is discarded). Used when the
        user starts talking over us: we cannot yet know if it was a request, "stop", or a cough. The
        caller must follow up with resume() or stop().
        """
        with self._lock:
            if self._paused:
                return
            self._paused = True
            self._paused_at = time.monotonic()
        trace.emit("speaker_pause", reason=reason)

    def resume(self, reason: str = "") -> None:
        with self._lock:
            was = self._paused
            self._paused = False
        if was:
            trace.emit("speaker_resume", reason=reason, held_s=round(time.monotonic() - self._paused_at, 2))

    def is_paused(self) -> bool:
        return self._paused

    def stop(self, reason: str = "") -> int:
        """Silence now: abort current item, discard queued ones, bump epoch."""
        with self._lock:
            self._paused = False
            self._duck = 1.0
            self.epoch += 1
            new_epoch = self.epoch
            cur = self._current
        self._abort.set()
        dropped = 0
        # Queued items are dropped lazily by the worker (epoch check); count them for the trace.
        dropped = self._q.qsize()
        trace.emit("speaker_stop", reason=reason, epoch=new_epoch,
                   current=(cur.kind + ":" + cur.label[:30]) if cur else None, queued=dropped)
        return new_epoch

    def set_duck(self, gain: float) -> None:
        """Target playback gain (<1 ducks our own voice while the user may be talking)."""
        self._duck = float(gain)

    def active_kind(self) -> Optional[str]:
        cur = self._current
        return cur.kind if cur else None

    def is_active(self) -> bool:
        return self._current is not None or self._pending > 0

    def wait_idle(self, timeout: Optional[float] = None) -> bool:
        return self._idle.wait(timeout)

    def recent_level(self, window_s: float = 0.35) -> float:
        """Max RMS (0..1) of audio written in the last `window_s` seconds."""
        cutoff = time.monotonic() - window_s
        vals = [r for (t, r) in list(self._levels) if t >= cutoff]
        return max(vals) if vals else 0.0

    def shutdown(self) -> None:
        self._running = False
        self.stop("shutdown")
        self._thread.join(timeout=2.0)

    # ── worker ────────────────────────────────────────────────────────────
    def _finish_dropped(self, item: AudioItem) -> None:
        item.completed = False
        item.done.set()
        if item.on_end:
            try:
                item.on_end(item)
            except Exception:
                log.exception("on_end callback")

    def _run(self) -> None:
        while self._running:
            try:
                item = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if item.epoch is not None and item.epoch < self.epoch:
                    self._finish_dropped(item)
                    continue
                self._play(item)
            except Exception:
                log.exception("playback failed")
                trace.emit("error", where="speaker", msg="playback failed")
                self._finish_dropped(item)
            finally:
                with self._lock:
                    self._pending -= 1
                    if self._pending <= 0:
                        self._pending = 0
                        self._idle.set()

    def _play(self, item: AudioItem) -> None:
        self._abort.clear()
        # Carry ducking/pausing across items: resetting to 1.0 here made every new sentence start at full
        # volume for a frame or two while the user was still talking over us.
        self._gain = 0.0 if self._paused else self._duck
        # An abort raised between the epoch check and here must still win.
        if item.epoch is not None and item.epoch < self.epoch:
            self._finish_dropped(item)
            return
        with self._lock:
            self._current = item
        item.started_at = time.monotonic()
        trace.emit("audio_start", kind=item.kind, label=item.label[:80], dur=round(item.duration_s, 3))
        if item.on_start:
            try:
                item.on_start(item)
            except Exception:
                log.exception("on_start callback")

        pcm, sr = item.pcm, item.sr
        play_item = item
        if self.backend.native_sr and sr != self.backend.native_sr:
            from scipy.signal import resample_poly
            pcm = (np.clip(resample_poly(pcm.astype(np.float32) / 32768.0, self.backend.native_sr, sr), -1, 1)
                   * 32767).astype(np.int16)
            sr = self.backend.native_sr
            play_item = type("ResampledItem", (), {"pcm": pcm})()
        chunk = max(1, int(sr * CHUNK_S))
        handle = self.backend.open(sr)
        offset, aborted, fade_left = 0, False, None
        try:
            while offset < len(pcm):
                if self._abort.is_set() or (item.epoch is not None and item.epoch < self.epoch):
                    if fade_left is None:
                        fade_left = 3              # ~60 ms linear fade-out, no click
                    if fade_left == 0:
                        aborted = True
                        break
                if self._paused and time.monotonic() - self._paused_at > self.pause_timeout_s:
                    log.warning("pause not resolved after %.0fs → resuming", self.pause_timeout_s)
                    self.resume("timeout")
                if self._paused and self._gain <= 0.001 and fade_left is None:
                    time.sleep(0.01)                  # faded out: hold position, write nothing
                    continue
                end = min(offset + chunk, len(pcm))
                seg = pcm[offset:end]
                target = 0.0 if self._paused else self._duck
                gains = (1.0, 1.0)
                if fade_left is not None:
                    g0, g1 = self._gain * fade_left / 3.0, self._gain * (fade_left - 1) / 3.0
                    seg = (seg.astype(np.float32) * np.linspace(g0, g1, len(seg), dtype=np.float32)).astype(np.int16)
                    gains = (g0, g1)
                    fade_left -= 1
                elif self._gain != target or target != 1.0:
                    seg = (seg.astype(np.float32) * np.linspace(self._gain, target, len(seg), dtype=np.float32)).astype(np.int16)
                    gains = (self._gain, target)
                    self._gain = target
                rms = float(np.sqrt(np.mean((seg.astype(np.float32) / 32768.0) ** 2))) if len(seg) else 0.0
                self._levels.append((time.monotonic(), rms))
                for tap in self.taps:
                    tap(item.started_at, offset, seg, sr, play_item, gains)
                self.backend.write(handle, seg.tobytes())
                offset = end
            item.completed = not aborted
            if item.completed and self.drain_extra_s > 0:
                time.sleep(self.drain_extra_s)
        finally:
            self.backend.close(handle, drain=not aborted)
            item.ended_at = time.monotonic()
            self.last_end = item.ended_at
            with self._lock:
                self._current = None
            trace.emit("audio_end", kind=item.kind, label=item.label[:80], completed=item.completed,
                       played_s=round(item.ended_at - item.started_at, 3))
            item.done.set()
            if item.on_end:
                try:
                    item.on_end(item)
                except Exception:
                    log.exception("on_end callback")
