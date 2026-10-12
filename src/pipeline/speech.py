"""
SpeechStreamer — turns text chunks into audio on the speaker while the LLM is
still generating. A worker thread synthesises chunk N+1 via the TTS server while
chunk N is playing, so there is no gap between sentences once speech has started.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import List, Optional

from audio.speaker import AudioItem, Speaker, pcm_from_wav_bytes
from common.log import get_logger, trace

log = get_logger("speech")


class SpeechStreamer:
    def __init__(self, tts, speaker: Speaker, epoch: int, turn: int):
        self.tts = tts
        self.speaker = speaker
        self.epoch = epoch
        self.turn = turn
        self._q: "queue.Queue[Optional[str]]" = queue.Queue()
        self._items: List[AudioItem] = []
        self._lock = threading.Lock()
        self.first_audio = threading.Event()
        self.first_audio_t: Optional[float] = None
        self.first_submit_t: Optional[float] = None
        self.spoken: List[str] = []            # chunks whose playback started (in order)
        self.completed_chunks = 0
        self.failed = False
        self.synth_ms: List[float] = []
        self._finished = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"TTSStream{turn}", daemon=True)
        self._thread.start()

    # ── producer side ─────────────────────────────────────────────────────
    def submit(self, text: str) -> None:
        if self.first_submit_t is None:
            self.first_submit_t = time.monotonic()
        self._q.put(text)

    def finish(self) -> None:
        """No more text will be submitted."""
        self._q.put(None)

    def cancelled(self) -> bool:
        return self.speaker.epoch != self.epoch

    # ── worker ────────────────────────────────────────────────────────────
    def _run(self) -> None:
        try:
            while True:
                text = self._q.get()
                if text is None:
                    return
                if self.cancelled():
                    continue                       # drain; nothing more to say
                t0 = time.monotonic()
                wav = self.tts._synthesize(text)
                dt = (time.monotonic() - t0) * 1000
                self.synth_ms.append(dt)
                if wav is None:
                    self.failed = True
                    trace.emit("error", where="tts", msg="synthesis failed", turn=self.turn)
                    return
                if self.cancelled():
                    continue
                try:
                    pcm, sr = pcm_from_wav_bytes(wav)
                except Exception as e:
                    self.failed = True
                    trace.emit("error", where="tts", msg=f"bad wav: {e}", turn=self.turn)
                    return
                trace.emit("tts_synth", turn=self.turn, ms=round(dt), chars=len(text), audio_s=round(len(pcm) / sr, 2))
                item = AudioItem("tts", pcm, sr, label=text, epoch=self.epoch,
                                 on_start=lambda it, tx=text: self._on_start(it, tx),
                                 on_end=self._on_end)
                with self._lock:
                    self._items.append(item)
                self.speaker.enqueue(item)
        finally:
            self._finished.set()

    def _on_start(self, item: AudioItem, text: str) -> None:
        self.spoken.append(text)
        if not self.first_audio.is_set():
            self.first_audio_t = time.monotonic()
            self.first_audio.set()
            trace.emit("tts_first_audio", turn=self.turn)

    def _on_end(self, item: AudioItem) -> None:
        if item.completed:
            self.completed_chunks += 1

    # ── consumer side ─────────────────────────────────────────────────────
    def wait(self, poll: float = 0.05) -> bool:
        """Block until everything submitted was played (True) or the turn was
        cancelled / failed (False)."""
        while not self._finished.wait(poll):
            if self.cancelled():
                return False
        with self._lock:
            items = list(self._items)
        for it in items:
            while not it.done.wait(poll):
                if self.cancelled() and self.speaker.active_kind() is None:
                    return False
        if self.failed:
            return False
        return not self.cancelled() and all(it.completed for it in items)
