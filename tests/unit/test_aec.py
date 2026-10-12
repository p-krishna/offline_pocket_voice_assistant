"""Echo canceller: converges on a synthetic room, keeps near-end speech, never blows up."""
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from audio.aec import EchoCanceller, ReferenceBuffer   # noqa: E402

SR, B = 16000, 480


def real_speech(seconds, skip=0):
    """Speech clips cached by the e2e suite (Kokoro TTS), else None."""
    import glob
    files = sorted(glob.glob(str(ROOT / "tests" / "e2e" / ".cache" / "*.npy")))
    xs = [np.load(f) for f in files]
    xs = [x for x in xs if len(x) > SR]
    if not xs:
        return None
    x = np.concatenate(xs[skip:] + xs[:skip])
    return (x[: int(seconds * SR)] * 0.5).astype(np.float32) if len(x) >= seconds * SR else None


def speechlike(seconds, seed):
    """Band-limited noise with syllable-rate amplitude modulation (stand-in for TTS speech)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    x = rng.normal(0, 1, n)
    x = np.convolve(x, np.hanning(24), mode="same")
    t = np.arange(n) / SR
    env = np.clip(np.sin(2 * np.pi * 3.7 * t) + 0.5 * np.sin(2 * np.pi * 1.3 * t + seed), 0, None)
    x = x * env
    return (0.25 * x / np.max(np.abs(x))).astype(np.float32)


def room(x, delay_s=0.09, gain=0.5, tail_s=0.05, seed=3):
    rng = np.random.default_rng(seed)
    h = np.zeros(int((delay_s + tail_s) * SR))
    d = int(delay_s * SR)
    h[d] = gain
    tail = rng.normal(0, 1, len(h) - d - 1) * np.exp(-np.arange(len(h) - d - 1) / (0.015 * SR))
    h[d + 1:] = 0.15 * gain * tail / np.max(np.abs(tail))
    return np.convolve(x, h)[: len(x)].astype(np.float32)


def run(aec, ref_signal, mic_signal, t0=1000.0):
    """Feed block by block; returns cleaned int16 signal."""
    out = []
    n = len(mic_signal) // B
    for i in range(n):
        tb = t0 + i * B / SR
        # the far-end block that played during this mic block (same timeline)
        aec.ref.push(tb, 0, (ref_signal[i * B:(i + 1) * B] * 32768).astype(np.int16), SR)
        mic = (np.clip(mic_signal[i * B:(i + 1) * B], -1, 1) * 32768).astype(np.int16)
        out.append(aec.process(mic, tb + B / SR))
    return np.concatenate(out).astype(np.float32) / 32768.0


def erle(mic, out, a, b):
    return 10 * np.log10((np.mean(mic[a:b] ** 2) + 1e-12) / (np.mean(out[a:b] ** 2) + 1e-12))


def test_converges_on_echo_only():
    x = real_speech(10) if real_speech(10) is not None else speechlike(16, 1)
    d = room(x) + np.random.default_rng(0).normal(0, 0.001, len(x)).astype(np.float32)
    aec = EchoCanceller(B, 14, 1.6, SR)
    out = run(aec, x, d)
    last = slice(int(6 * SR), len(x))
    e = erle(d, out, last.start, last.stop)
    assert e > 15, f"ERLE only {e:.1f} dB"


def test_near_end_speech_survives_double_talk():
    x = real_speech(14) if real_speech(14) is not None else speechlike(14, 1)
    echo = room(x)
    # converge first on echo only, then user talks (different signal) over the echo
    near = np.zeros_like(x)
    s = real_speech(3, skip=3) if real_speech(3, skip=3) is not None else speechlike(3, 7)
    s = s * 2.0
    near[int(8 * SR): int(8 * SR) + len(s)] = s
    d = echo + near
    aec = EchoCanceller(B, 14, 1.6, SR)
    out = run(aec, x, d)
    seg = slice(int(8.2 * SR), int(10.8 * SR))
    resid_echo = np.mean((out[seg] - near[seg]) ** 2)
    near_pow = np.mean(near[seg] ** 2)
    assert near_pow > 0
    # residual (echo left + distortion of the near-end) well below the near-end power
    assert resid_echo < 0.5 * near_pow, f"residual/near = {resid_echo / near_pow:.2f}"
    # the canceller must not make the echo problem worse than doing nothing
    assert resid_echo < 0.25 * np.mean(echo[seg] ** 2) + 0.01 * near_pow, 'echo not suppressed / near-end distorted'


def test_passthrough_when_nothing_is_playing():
    aec = EchoCanceller(B, 8, 1.6, SR)
    mic = (np.random.default_rng(1).normal(0, 0.05, B) * 32768).astype(np.int16)
    out = aec.process(mic, 2000.0)
    assert np.array_equal(out, mic) and not aec.active


def test_realtime_cost_is_small():
    aec = EchoCanceller(B, 14, 1.6, SR)
    x = speechlike(2, 1)
    d = room(x)
    t = time.perf_counter()
    run(aec, x, d)
    per_block_ms = (time.perf_counter() - t) / (len(x) // B) * 1000
    assert per_block_ms < 3.0, f"{per_block_ms:.2f} ms per 30 ms block"


def test_reference_buffer_gap_is_silence():
    r = ReferenceBuffer(SR, 5)
    r.push(10.0, 0, np.full(1600, 1000, dtype=np.int16), SR)
    r.push(10.5, 0, np.full(1600, 2000, dtype=np.int16), SR)
    a = r.read(10.0, 1600)
    gap = r.read(10.2, 1600)
    b = r.read(10.5, 1600)
    assert abs(a.mean() - 1000 / 32768) < 1e-6 and np.abs(gap).max() == 0 and abs(b.mean() - 2000 / 32768) < 1e-6


def test_bulk_delay_is_found_and_speeds_convergence():
    rng = np.random.default_rng(5)
    x = (rng.normal(0, 0.1, SR * 4)).astype(np.float32)           # calibration-style noise
    d = room(x, delay_s=0.27, gain=0.5) + rng.normal(0, 0.001, len(x)).astype(np.float32)
    aec = EchoCanceller(B, 14, 1.6, SR)
    out = run(aec, x, d)
    assert aec.delay_locked and abs(aec.ref_delay_s - (0.27 - 0.06)) < 0.04, aec.ref_delay_s
    assert erle(d, out, int(2.5 * SR), len(x)) > 12


def test_no_echo_path_does_not_lock_or_distort():
    rng = np.random.default_rng(6)
    x = (rng.normal(0, 0.1, SR * 3)).astype(np.float32)
    d = rng.normal(0, 0.003, len(x)).astype(np.float32)           # headset: nothing leaks into the mic
    aec = EchoCanceller(B, 14, 1.6, SR)
    out = run(aec, x, d)
    assert not aec.delay_locked and aec.bypassed
    tail = slice(2 * SR, None)
    assert np.allclose(out[tail], d[tail], atol=2 / 32768)          # untouched once it knows there is no echo path


def test_queued_reference_models_underrun_gaps_and_gapless_playback():
    class Item:
        pcm = np.full(4800, 1000, dtype=np.int16)

    r = ReferenceBuffer(SR, 5)
    it = Item()
    r.push_queued(it, 0, 1600, SR)             # starts now
    r.push_queued(it, 1600, 1600, SR)          # writer is ahead → continues gaplessly
    t0 = r.t0
    assert abs(r.read(t0, 3200).mean() - 1000 / 32768) < 1e-6
    time.sleep(0.4)                            # writer stalls: device runs dry
    r.push_queued(it, 3200, 1600, SR)          # resumes later → audible gap in the reference
    gap = r.read(t0 + 0.25, 1600)              # 0.20 s of audio played, then silence
    assert np.abs(gap).max() == 0
    assert abs(r.read(t0 + 0.42, 800).mean() - 1000 / 32768) < 1e-6


# ── review findings F12 / F13 ────────────────────────────────────────────────
def test_trusted_is_latched_and_cleared_on_reset():
    from audio.aec import EchoCanceller
    e = EchoCanceller(trust_db=8.0)
    assert not e.trusted
    e.converged, e._trusted = True, True
    e.erle_db = 2.0                      # a quiet spell: smoothed ERLE dips, trust must not flip
    assert e.trusted
    e.converged = False                  # filter reset
    assert not e.trusted


def test_trust_is_reached_through_processing():
    import numpy as np
    from audio.aec import EchoCanceller, ReferenceBuffer
    ref = ReferenceBuffer(16000)
    e = EchoCanceller(block=480, partitions=4, ref=ref, trust_db=8.0)
    rng = np.random.default_rng(1)
    t = 50.0
    far = (rng.standard_normal(480 * 400) * 3000).astype(np.int16)
    for k in range(400):
        seg = far[k * 480:(k + 1) * 480]
        ref.push(t + k * 0.03, 0, seg, 16000)
        # the microphone hears the far end 30 ms later, attenuated
        e.process((seg * 0.4).astype(np.int16), t + (k + 1) * 0.03 + 0.03)
    assert e.trusted == (e.converged and e.erle_db >= 0 and e._trusted)


def test_delay_estimation_is_rearmed_after_giving_up():
    import numpy as np
    from audio.aec import EchoCanceller, ReferenceBuffer
    ref = ReferenceBuffer(16000)
    e = EchoCanceller(block=480, partitions=4, ref=ref)
    e.delay_tries = 8                       # gave up earlier (headset)
    e.delay_locked = False
    t = 100.0
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(480) * 3000).astype(np.int16)
    for k in range(700):                    # 700 blocks with far-end audio present
        ref.push(t + k * 0.03, 0, x, 16000)
        e.process(np.zeros(480, dtype=np.int16), t + (k + 1) * 0.03)
    assert e.delay_tries < 8, "estimation must be retried after a long stretch of far-end audio"
