"""
Software acoustic echo cancellation (AEC).

The assistant knows exactly what it is playing, so what the microphone hears of
it can be predicted and subtracted. Without this, the assistant's own voice looks
like "someone is talking" and it interrupts itself on any loudspeaker setup.

Algorithm: partitioned-block frequency-domain adaptive filter (PBFDAF, overlap-save)
with per-bin normalisation, adaptation freezing during probable double-talk, and
a divergence guard. Block size == mic frame (30 ms @ 16 kHz); K partitions give a
filter spanning K*30 ms, which must cover output-device latency + room echo path.

Pieces
  ReferenceBuffer  timeline of what was sent to the loudspeaker (16 kHz)
  EchoCanceller    process(mic_block, t_block_end) -> cleaned block
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
from scipy.signal import resample_poly


class ReferenceBuffer:
    """Thread-safe 16 kHz record of loudspeaker output indexed by wall-clock time."""

    def __init__(self, sr: int = 16000, seconds: float = 20.0):
        self.sr = sr
        self.size = int(sr * seconds)
        self.buf = np.zeros(self.size, dtype=np.float32)
        self.t0: Optional[float] = None
        self.end_idx = 0                # first sample index never written
        self._cur_item = None
        self._cur_ref = None
        self._clk = 0.0                 # device-queue clock (monotonic seconds) for push_queued
        self.lock = threading.Lock()

    def _idx(self, t: float) -> int:
        return int(round((t - self.t0) * self.sr))

    def push(self, t_item_start: float, offset: int, pcm_int16: np.ndarray, src_sr: int,
             item=None, gain=(1.0, 1.0)) -> None:
        """
        A chunk of loudspeaker output starting `offset` samples (at src_sr) into an item that began
        playing at monotonic `t_item_start`. When `item` is given the *whole* item is resampled once
        (chunk-wise resampling would add boundary artifacts the real loudspeaker does not have).
        """
        n_src = len(pcm_int16)
        if item is not None and src_sr != self.sr:
            if self._cur_item is not item:
                self._cur_item = item
                self._cur_ref = resample_poly(item.pcm.astype(np.float32) / 32768.0, self.sr, src_sr).astype(np.float32)
            a0 = int(round(offset * self.sr / src_sr))
            a1 = int(round((offset + n_src) * self.sr / src_sr))
            x = self._cur_ref[a0:a1].copy()
            offset = a0
        elif item is not None:
            x = item.pcm[offset:offset + n_src].astype(np.float32) / 32768.0
        else:
            x = pcm_int16.astype(np.float32) / 32768.0
            if src_sr != self.sr:
                x = resample_poly(x, self.sr, src_sr).astype(np.float32)
                offset = int(round(offset * self.sr / src_sr))
        if len(x) and gain != (1.0, 1.0):
            x = x * np.linspace(gain[0], gain[1], len(x), dtype=np.float32)
        with self.lock:
            if self.t0 is None:
                self.t0 = t_item_start
            a = self._idx(t_item_start) + offset
            if a < 0:
                return
            b = a + len(x)
            if a > self.end_idx:                       # silence between items
                self._zero(self.end_idx, a)
            for i0, i1, j0 in self._spans(a, b):
                self.buf[i0:i1] = x[j0:j0 + (i1 - i0)]
            self.end_idx = max(self.end_idx, b)

    def push_queued(self, item, offset: int, n_src: int, src_sr: int, gain=(1.0, 1.0)) -> None:
        """
        Model the output device as a gapless queue: a chunk handed to the device starts playing when the
        previous one ends, or immediately if the device ran dry (writer was late → audible gap, exactly
        what an underrun does). Call synchronously, right before the blocking device write. The constant
        device latency is absorbed by the echo canceller's delay lock.
        """
        now = time.monotonic()
        if item is not self._cur_item:
            self._cur_item = item
            x_all = item.pcm.astype(np.float32) / 32768.0
            self._cur_ref = resample_poly(x_all, self.sr, src_sr).astype(np.float32) if src_sr != self.sr else x_all
        a0 = int(round(offset * self.sr / src_sr))
        a1 = int(round((offset + n_src) * self.sr / src_sr))
        x = self._cur_ref[a0:a1].copy()
        if len(x) and gain != (1.0, 1.0):
            x = x * np.linspace(gain[0], gain[1], len(x), dtype=np.float32)
        with self.lock:
            if self.t0 is None:
                self.t0 = now
            start = max(now, self._clk)
            self._clk = start + len(x) / self.sr
            a = self._idx(start)
            if a < 0:
                return
            b = a + len(x)
            if a > self.end_idx:
                self._zero(self.end_idx, a)
            for i0, i1, j0 in self._spans(a, b):
                self.buf[i0:i1] = x[j0:j0 + (i1 - i0)]
            self.end_idx = max(self.end_idx, b)

    def _zero(self, a: int, b: int) -> None:
        a = max(a, b - self.size)
        for i0, i1, _ in self._spans(a, b):
            self.buf[i0:i1] = 0.0

    def _spans(self, a: int, b: int):
        """Map absolute [a,b) onto the ring → (ring_start, ring_end, offset_in_source)."""
        j = 0
        while a < b:
            r = a % self.size
            n = min(b - a, self.size - r)
            yield r, r + n, j
            a += n
            j += n

    def read(self, t_start: float, n: int) -> np.ndarray:
        """Reference samples for [t_start, t_start+n/sr); zeros where nothing was played."""
        out = np.zeros(n, dtype=np.float32)
        with self.lock:
            if self.t0 is None:
                return out
            a = self._idx(t_start)
            lo = max(a, self.end_idx - self.size, 0)
            hi = min(a + n, self.end_idx)
            if hi > lo:
                got = np.concatenate([self.buf[i0:i1] for i0, i1, _ in self._spans(lo, hi)])
                out[lo - a: hi - a] = got
        return out


class EchoCanceller:
    def __init__(self, block: int = 480, partitions: int = 12, mu: float = 1.6, sr: int = 16000,
                 ref: Optional[ReferenceBuffer] = None, ref_delay_s: float = 0.0):
        self.B = block
        self.K = partitions
        self.mu = mu
        self.sr = sr
        self.ref = ref or ReferenceBuffer(sr)
        self.ref_delay_s = ref_delay_s
        N = 2 * block
        self.N = N
        self.W = np.zeros((partitions, block + 1), dtype=np.complex64)
        self.X = np.zeros((partitions, block + 1), dtype=np.complex64)
        self.xbuf = np.zeros(N, dtype=np.float32)
        self.far_energy_hist = np.zeros(partitions, dtype=np.float32)
        self.erle_db = 0.0              # smoothed echo return loss enhancement
        self._bad = 0
        self.active = False             # far-end present in the filter span during the last block
        self.blocks = 0
        self.resets = 0
        self.trained_blocks = 0
        self.dtd_run = 0
        self.hang = 0
        self.converged = False
        # bulk-delay estimation (GCC-PHAT) so the filter span is centred on the real echo path
        self.delay_locked = False
        self.delay_hist = []
        self.delay_conf = 0.0
        self.delay_tries = 0
        self.bypassed = False

    def _estimate_delay(self) -> None:
        """GCC-PHAT between the recent mic and reference signals → bulk echo delay."""
        mic = np.concatenate([m for m, _ in self.delay_hist])
        ref = np.concatenate([r for _, r in self.delay_hist])
        n = 1 << int(np.ceil(np.log2(len(mic) + len(ref))))
        M, R = np.fft.rfft(mic, n), np.fft.rfft(ref, n)
        cross = M * np.conj(R)
        cc = np.fft.irfft(cross / (np.abs(cross) + 1e-9), n)
        max_lag = int(0.6 * self.sr)
        c = cc[: max_lag + 1]               # mic lags the reference by 0..600 ms
        k = int(np.argmax(c))
        conf = float(c[k] / (np.median(np.abs(c)) + 1e-9))
        self.delay_conf = conf
        self.delay_tries += 1
        if conf > 6.0:
            self.ref_delay_s = max(0.0, k / self.sr - 0.06)
            self.delay_locked = True
            self.W[:] = 0
            self.converged = False
            self.erle_db = 0.0
            self.trained_blocks = 0
            self.X[:] = 0
            self.xbuf[:] = 0
        self.delay_hist = []

    # ── public ────────────────────────────────────────────────────────────
    def process(self, mic_block: np.ndarray, t_block_end: float) -> np.ndarray:
        """mic_block: int16 (B,). Returns int16 (B,) with the predicted echo removed."""
        B = self.B
        t0 = t_block_end - B / self.sr - self.ref_delay_s
        x = self.ref.read(t0, B)
        fe = float(np.dot(x, x)) / B
        self.far_energy_hist = np.roll(self.far_energy_hist, 1)
        self.far_energy_hist[0] = fe
        self.blocks += 1
        # Slide the far-end spectra even when silent so partitions stay time-aligned.
        self.xbuf = np.concatenate([self.xbuf[B:], x])
        Xn = np.fft.rfft(self.xbuf).astype(np.complex64)
        self.X = np.roll(self.X, 1, axis=0)
        self.X[0] = Xn
        self.active = bool(self.far_energy_hist.max() > 1e-7)
        if not self.active:
            return mic_block

        if self.delay_locked and not self.converged and self.trained_blocks > 300:
            # ~9 s of far-end audio and still no cancellation: the echo path probably changed
            # (speaker moved, volume/device switched). Start over, including delay estimation.
            self.delay_locked = False
            self.delay_tries = 0
            self.ref_delay_s = 0.0
            self.W[:] = 0
            self.trained_blocks = 0
            self.delay_hist = []

        if not self.delay_locked and self.delay_tries < 8:
            # Collect ~1 s of far-end-active audio with the reference read WITHOUT any delay.
            if fe > 1e-5:
                xr = self.ref.read(t_block_end - B / self.sr, B) if self.ref_delay_s else x
                self.delay_hist.append((mic_block.astype(np.float32) / 32768.0, xr))
                if len(self.delay_hist) >= 32:
                    self._estimate_delay()
            else:
                self.delay_hist = []              # need an unbroken stretch of far-end audio

        if not self.delay_locked and self.delay_tries >= 1:
            # No echo path found (headset / speakers far away or off): leave the signal untouched
            # rather than let an adaptive filter chase noise. Estimation keeps retrying.
            self.bypassed = True
            return mic_block
        self.bypassed = False

        d = mic_block.astype(np.float32) / 32768.0
        Y = np.sum(self.W * self.X, axis=0)
        y = np.fft.irfft(Y, self.N).astype(np.float32)[B:]
        e = d - y

        Ed = float(np.dot(d, d)) / B + 1e-12
        Ee = float(np.dot(e, e)) / B + 1e-12
        Ey = float(np.dot(y, y)) / B + 1e-12

        # Divergence guard: filter making things worse repeatedly → reset it.
        if Ee > 1.5 * Ed and fe > 1e-6:
            self._bad += 1
            if self._bad >= 6:
                self.W[:] = 0
                self.converged = False
                self.erle_db = 0.0
                self._bad = 0
                self.resets += 1
                return mic_block
        else:
            self._bad = 0

        # Double-talk handling. Once the filter has converged, a mic clearly louder than the echo
        # it predicts means the user is talking: do not let the filter learn their voice.
        # Hang-over keeps adaptation off through the quiet moments between their words; the
        # freeze is lifted after ~3 s so a genuine echo-path change is still tracked.
        near = self.converged and Ed > 3.0 * Ey + 1e-5
        if near:
            self.hang = 10
        dtd = self.hang > 0
        if self.hang > 0 and not near:
            self.hang -= 1
        self.dtd_run = self.dtd_run + 1 if dtd else 0
        mu = 0.0 if (dtd and self.dtd_run < 100) else self.mu
        if fe > 1e-5:
            self.trained_blocks += 1
            if not dtd:
                self.erle_db = 0.95 * self.erle_db + 0.05 * float(np.clip(10 * np.log10(Ed / Ee), -10, 40))
                if self.erle_db > 8.0:
                    self.converged = True
        if mu > 0 and fe > 1e-7:
            E = np.fft.rfft(np.concatenate([np.zeros(B, dtype=np.float32), e])).astype(np.complex64)
            P = np.sum(np.abs(self.X) ** 2, axis=0) + 1e-3 * B
            G = np.conj(self.X) * (E / P)[None, :]
            g = np.fft.irfft(G, self.N, axis=1)[:, :B]
            g = np.concatenate([g, np.zeros_like(g)], axis=1)
            self.W += (mu * np.fft.rfft(g, axis=1)).astype(np.complex64)
        out = np.clip(e, -1.0, 1.0)
        return (out * 32768.0).astype(np.int16)
