"""Earcon synthesis (pure numpy, no I/O)."""
from __future__ import annotations

import numpy as np


def make_earcon(tones, samplerate: int, amplitude: int = 13000, fade_ms: float = 5.0) -> np.ndarray:
    """
    Build a multi-tone earcon from [(freq_hz, duration_s), ...].
    First tone fades in and the last fades out so it never clicks.
    Returns int16 mono samples.
    """
    segments = []
    fade = int(samplerate * fade_ms / 1000)
    for i, (freq, dur) in enumerate(tones):
        n = int(samplerate * dur)
        t = np.linspace(0, dur, n, endpoint=False)
        wave = np.sin(2 * np.pi * freq * t)
        if i == 0 and fade > 0:
            wave[:fade] *= np.linspace(0, 1, fade)
        if i == len(tones) - 1 and fade > 0:
            wave[-fade:] *= np.linspace(1, 0, fade)
        segments.append(wave)
    return (np.concatenate(segments) * amplitude).astype(np.int16)


def build_earcons(sr: int) -> dict:
    """Named earcon set used by the pipeline."""
    return {
        # Wake word accepted: rising two-tone ("I'm listening").
        "wake": make_earcon([(660, 0.08), (880, 0.08)], sr),
        # Utterance captured: one very short soft tick ("got it, working").
        "ack": make_earcon([(988, 0.06)], sr, amplitude=7000),
        # Still working: gentle low pulse, clearly not an alert.
        "pulse": make_earcon([(587, 0.07), (659, 0.07)], sr, amplitude=5000),
        # Barge-in registered: falling two-tone.
        "interrupt": make_earcon([(880, 0.06), (660, 0.06)], sr, amplitude=9000),
        # Something went wrong.
        "error": make_earcon([(440, 0.10), (311, 0.10), (220, 0.12)], sr, amplitude=11000),
        # Going to sleep.
        "sleep": make_earcon([(880, 0.10), (523, 0.10), (330, 0.12)], sr, amplitude=10000),
    }


def make_calibration(samplerate: int, seconds: float = 3.0, rms: float = 0.04, seed: int = 7) -> np.ndarray:
    """
    Soft speech-band noise with a slow random envelope: excites the echo canceller across
    the frequencies that matter, so it is already trained before the first spoken reply.
    """
    rng = np.random.default_rng(seed)
    n = int(samplerate * seconds)
    x = rng.normal(0, 1, n)
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / samplerate)
    spec[(f < 150) | (f > 5000)] = 0
    x = np.fft.irfft(spec, n)
    env = 0.6 + 0.4 * np.sin(2 * np.pi * np.cumsum(rng.uniform(2.0, 5.0, n)) / samplerate)
    fade = int(0.25 * samplerate)
    env[:fade] *= np.linspace(0, 1, fade)
    env[-fade:] *= np.linspace(1, 0, fade)
    x = x * env
    x *= rms / (float(np.sqrt(np.mean(x ** 2))) + 1e-9)
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)
