"""Speaker: pause/resume/stop/duck semantics (F11, F22) against a fake output device."""
import threading
import time

import numpy as np

from audio.speaker import AudioItem, Speaker, SpeakerBackend


class FakeBackend(SpeakerBackend):
    def __init__(self, realtime=True):
        self.written = []          # (monotonic, int16 array)
        self.realtime = realtime
        self.lock = threading.Lock()

    def open(self, sr):
        return object()

    def write(self, handle, pcm_bytes):
        a = np.frombuffer(pcm_bytes, dtype=np.int16).copy()
        with self.lock:
            self.written.append((time.monotonic(), a))
        if self.realtime:
            time.sleep(len(a) / 16000.0)

    def close(self, handle, drain):
        pass

    def total(self):
        with self.lock:
            return sum(len(a) for _, a in self.written)


def tone(seconds, sr=16000, amp=8000):
    t = np.arange(int(seconds * sr)) / sr
    return (np.sin(2 * np.pi * 440 * t) * amp).astype(np.int16)


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_plays_item_to_completion():
    be = FakeBackend()
    sp = Speaker(be, drain_extra_s=0)
    item = sp.enqueue(AudioItem("tts", tone(0.3), 16000, label="a"))
    assert item.done.wait(3)
    assert item.completed
    assert be.total() == len(item.pcm)
    sp.shutdown()


def test_pause_holds_position_and_resume_finishes_the_item():
    be = FakeBackend()
    sp = Speaker(be, drain_extra_s=0)
    item = sp.enqueue(AudioItem("tts", tone(1.0), 16000, label="long"))
    assert wait_for(lambda: be.total() > 3000)
    sp.pause("test")
    time.sleep(0.2)                       # fade-out completes
    n = be.total()
    time.sleep(0.3)
    assert be.total() == n, "nothing may be written while paused"
    assert not item.done.is_set(), "a paused item is held, not finished"
    sp.resume("test")
    assert item.done.wait(3)
    assert item.completed, "after resume the whole item is played"
    assert be.total() == len(item.pcm)
    sp.shutdown()


def test_paused_audio_fades_out_instead_of_clicking():
    be = FakeBackend()
    sp = Speaker(be, drain_extra_s=0)
    sp.enqueue(AudioItem("tts", tone(1.0), 16000, label="long"))
    assert wait_for(lambda: be.total() > 3000)
    sp.pause("test")
    time.sleep(0.3)
    with be.lock:
        audio = np.concatenate([a for _, a in be.written])
    assert abs(int(audio[-1])) < 50, "the last written sample must be ~0 (no click at the hold)"
    ramp = np.abs(audio[-320:])             # the final 20 ms chunk is a linear ramp from the pre-pause gain to 0
    assert ramp[:80].max() > ramp[-80:].max()
    sp.stop("cleanup")
    sp.shutdown()


def test_stop_while_paused_discards_everything():
    be = FakeBackend()
    sp = Speaker(be, drain_extra_s=0)
    a = sp.enqueue(AudioItem("tts", tone(1.0), 16000, label="a"))
    b = sp.enqueue(AudioItem("tts", tone(1.0), 16000, label="b"))
    assert wait_for(lambda: be.total() > 3000)
    sp.pause("test")
    time.sleep(0.15)
    e0 = sp.epoch
    sp.stop("test")
    assert sp.epoch == e0 + 1
    assert not sp.is_paused()
    assert a.done.wait(3) and b.done.wait(3)
    assert not a.completed and not b.completed
    sp.shutdown()


def test_unresolved_pause_times_out_and_continues():
    be = FakeBackend()
    sp = Speaker(be, drain_extra_s=0, pause_timeout_s=0.4)
    item = sp.enqueue(AudioItem("tts", tone(0.6), 16000, label="a"))
    assert wait_for(lambda: be.total() > 1000)
    sp.pause("test")
    assert item.done.wait(4), "an unresolved pause must not mute the assistant for good"
    assert item.completed
    sp.shutdown()


def test_duck_survives_into_the_next_item():
    be = FakeBackend()
    sp = Speaker(be, drain_extra_s=0)
    sp.set_duck(0.25)
    first = sp.enqueue(AudioItem("tts", tone(0.2), 16000, label="a"))
    second = sp.enqueue(AudioItem("tts", tone(0.2), 16000, label="b"))
    assert second.done.wait(3)
    with be.lock:
        audio = np.concatenate([a for _, a in be.written])
    # second item starts after the first 3200 samples; its start must already be ducked (<= ~0.3 * 8000)
    start = audio[3200 + 100:3200 + 700]
    assert np.abs(start).max() < 0.35 * 8000
    sp.shutdown()


def test_stale_epoch_items_are_dropped():
    be = FakeBackend()
    sp = Speaker(be, drain_extra_s=0)
    old_epoch = sp.epoch
    sp.stop("bump")
    item = sp.enqueue(AudioItem("tts", tone(0.1), 16000, label="stale", epoch=old_epoch))
    assert item.done.wait(1)
    assert not item.completed
    assert be.total() == 0
    sp.shutdown()
