"""
src/common/event_logger.py

Thread-safe, append-only JSONL event logger.
Every pipeline module imports the shared `logger` singleton and calls it.

Log file: /tmp/assistant_events.jsonl
Each line: {"ts": <epoch_ms>, "kind": "<type>", "value": <float|null>, "label": <str|null>}

kinds used:
  amplitude      — RMS sample every 100 ms
  webrtc_speech / webrtc_silence
  wakeword       — value = confidence score
  silero_speech / silero_silence — value = avg probability
  stt_start / stt_result         — label = transcript on result
  llm_start / llm_result
  tts_start / tts_end
  utterance_accepted / utterance_rejected
  interrupt
  error          — label = message
"""

import json
import threading
import time
from pathlib import Path

LOG_PATH = Path("/tmp/assistant_events.jsonl")


class EventLogger:
    def __init__(self, log_path: Path = LOG_PATH):
        self._path = log_path
        self._lock = threading.Lock()
        # Clear the log at each session start so the visualizer always
        # shows data from the current run only.
        self._path.write_text("")

    def log(self, kind: str, value: float = None, label: str = None):
        entry = {"ts": int(time.time() * 1000), "kind": kind}
        if value is not None:
            entry["value"] = str(round(value, 5))
        if label is not None:
            entry["label"] = label[:200]
        with self._lock:
            with self._path.open("a") as fh:
                fh.write(json.dumps(entry, separators=(",", ":")) + "\n")

    # ── convenience helpers ───────────────────────────────────────────────
    def amplitude(self, rms: float):          self.log("amplitude", value=rms)
    def webrtc(self, is_speech: bool):        self.log("webrtc_speech" if is_speech else "webrtc_silence")
    def wakeword(self, score: float):         self.log("wakeword", value=score)
    def silero(self, is_speech: bool, avg: float = None):
        self.log("silero_speech" if is_speech else "silero_silence", value=avg)
    def stt_start(self):                      self.log("stt_start")
    def stt_result(self, transcript: str):    self.log("stt_result", label=transcript)
    def llm_start(self):                      self.log("llm_start")
    def llm_result(self):                     self.log("llm_result")
    def tts_start(self):                      self.log("tts_start")
    def tts_end(self):                        self.log("tts_end")
    def utterance_accepted(self, rms: float): self.log("utterance_accepted", value=rms)
    def utterance_rejected(self, rms: float): self.log("utterance_rejected", value=rms)
    def interrupt(self, rms: float):          self.log("interrupt", value=rms)
    def error(self, msg: str):                self.log("error", label=msg)


# Shared singleton — all modules import this one instance
logger = EventLogger()