"""Microphone sources. The pipeline only needs `read(n_samples) -> bytes` (int16 mono)."""
from __future__ import annotations

import time

import numpy as np
import pyaudio

from audio.devices import MicError, resolve_from_cfg
from common.log import get_logger

log = get_logger("mic")


class MicBase:
    sample_rate = 16000
    # Optional hook: called with "lost" / "restored" from the thread that calls read().
    on_status = None

    def open(self) -> None: ...
    def read(self, n_samples: int) -> bytes: raise NotImplementedError
    def close(self) -> None: ...


class PyAudioMic(MicBase):
    def __init__(self, cfg, frame_samples: int):
        self.cfg = cfg
        self.sample_rate = cfg.sample_rate
        self.frame_samples = frame_samples
        self.device = None
        self._pa = None
        self._stream = None
        self._frames = 0
        self._errors = 0
        self._consecutive = 0
        self._lost = False
        self._next_reopen = 0.0
        self._t_log = 0.0
        self._peak = 0
        self._sumsq = 0.0
        self._n = 0

    def open(self) -> None:
        self._pa = pyaudio.PyAudio()
        try:
            self.device = resolve_from_cfg(self._pa, self.cfg)
            log.info("opening stream: device=[%d] %s channels=1 rate=%d Hz (native %.0f) format=int16 frame=%d",
                     self.device.index, self.device.name, self.sample_rate,
                     self.device.default_rate, self.frame_samples)
            self._stream = self._pa.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate,
                                         input=True, input_device_index=self.device.index,
                                         frames_per_buffer=self.frame_samples)
        except Exception as e:
            self._pa.terminate()
            self._pa = None
            raise MicError(f"cannot open microphone: {e}") from e
        log.info("MICROPHONE OPEN: [%d] %s @ %d Hz mono", self.device.index, self.device.name, self.sample_rate)
        self._t_log = time.monotonic()

    def read(self, n_samples: int) -> bytes:
        try:
            data = self._stream.read(n_samples, exception_on_overflow=False)
        except (OSError, AttributeError) as e:
            # Device hiccup (e.g. USB mic unplugged / re-enumerated). Return silence so the loop keeps
            # running, and keep trying to get the device back (by name, so a new index is fine).
            self._errors += 1
            self._consecutive += 1
            if self._errors <= 3 or self._errors % 100 == 0:
                log.error("mic read failed (#%d) on %s: %s", self._errors,
                          self.device.name if self.device else "?", e)
            if self._consecutive >= 5 and not self._lost:
                self._lost = True
                self._notify("lost")
            if self._consecutive >= 5 and time.monotonic() >= self._next_reopen:
                self._next_reopen = time.monotonic() + 2.0
                self._try_reopen()
            time.sleep(n_samples / self.sample_rate)
            return b"\x00\x00" * n_samples
        self._consecutive = 0
        if self._lost:
            self._lost = False
            self._notify("restored")
        self._level_log(data)
        return data

    def _notify(self, event: str) -> None:
        log.warning("microphone %s", event)
        cb = self.on_status
        if cb is not None:
            try:
                cb(event)
            except Exception:
                log.exception("mic status callback")

    def _try_reopen(self) -> None:
        try:
            self.close()
        except Exception:
            pass
        try:
            self.open()
            log.info("microphone reopened")
        except Exception as e:
            self._stream = None
            log.debug("microphone reopen failed: %s", e)

    def _level_log(self, data: bytes) -> None:
        """Every 5 s at DEBUG: input level, so a dead/wrong mic is obvious (peak stays ~0)."""
        a = np.frombuffer(data, np.int16)
        if a.size:
            self._peak = max(self._peak, int(np.abs(a).max()))
            self._sumsq += float((a.astype(np.float64) ** 2).sum())
            self._n += a.size
        now = time.monotonic()
        if now - self._t_log >= 5.0 and self._n:
            rms = (self._sumsq / self._n) ** 0.5
            lvl = log.warning if self._peak < 20 else log.debug
            lvl("mic level (5 s) on %s: rms=%.0f peak=%d%s", self.device.name, rms, self._peak,
                "  <-- near-silence: wrong device, muted, or unplugged?" if self._peak < 20 else "")
            self._t_log, self._peak, self._sumsq, self._n = now, 0, 0.0, 0
    def close(self) -> None:
        try:
            if self._stream:
                self._stream.stop_stream()
                self._stream.close()
            if self._pa:
                self._pa.terminate()
        except Exception as e:  # pragma: no cover
            log.warning("mic close: %s", e)
