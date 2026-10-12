"""
src/common/log.py

Two complementary observability channels:

1. `get_logger(name)` — human-readable console/file logging with millisecond
   timestamps and the thread name (DEBUG / INFO / WARNING / ERROR). Level from
   LOG_LEVEL (default INFO). Set LOG_FILE to also mirror to a file.

2. `trace` — a thread-safe, structured JSONL trace of the *conversation*:
   state transitions, audio start/end, STT/LLM/TTS timings, barge-ins, errors.
   One JSON object per line:
       {"t": <seconds since process start>, "ts": <epoch ms>, "ev": "<name>", ...fields}
   The automated end-to-end evaluator (tests/e2e) reads this trace to judge
   timing and behaviour, and it is just as useful when debugging a live session.
   Path: TRACE_PATH env var, else debug_logs/trace_<timestamp>.jsonl.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, List, Optional

_T0 = time.monotonic()
_configured = False
_cfg_lock = threading.Lock()


class _Fmt(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ms = int((record.created % 1) * 1000)
        ts = time.strftime("%H:%M:%S", time.localtime(record.created)) + f".{ms:03d}"
        th = record.threadName.replace("Thread", "")[:10]
        msg = record.getMessage()
        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)
        return f"{ts} {record.levelname[0]} [{th:<10}] {record.name.split('.')[-1]:<8} {msg}"


def configure_logging(level: Optional[str] = None) -> None:
    global _configured
    with _cfg_lock:
        if _configured:
            return
        lvl = getattr(logging, (level or os.getenv("LOG_LEVEL", "INFO")).upper(), logging.INFO)
        root = logging.getLogger("assistant")
        root.setLevel(lvl)
        root.propagate = False
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(_Fmt())
        root.addHandler(h)
        log_file = os.getenv("LOG_FILE")
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(_Fmt())
            root.addHandler(fh)
        _configured = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(f"assistant.{name}")


class Tracer:
    """Append-only structured event trace (see module docstring)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._fh = None
        self._path: Optional[Path] = None
        self.events: List[dict] = []          # in-memory copy for in-process evaluators
        self._listeners: List[Callable[[dict], None]] = []
        self.keep_in_memory = True

    @property
    def path(self) -> Optional[Path]:
        return self._path

    def open(self, path: Optional[str] = None) -> Path:
        """(Re)open the trace file. Safe to call again to start a fresh trace."""
        with self._lock:
            if self._fh:
                try:
                    self._fh.close()
                except Exception:
                    pass
            if path is None:
                path = os.getenv("TRACE_PATH") or str(
                    Path(os.getenv("TURN_LOG_DIR", "debug_logs"))
                    / f"trace_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
                )
            self._path = Path(path)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self._path, "w", buffering=1)
            self.events = []
            return self._path

    def add_listener(self, fn: Callable[[dict], None]) -> None:
        self._listeners.append(fn)

    def emit(self, ev: str, **fields: Any) -> dict:
        rec = {"t": round(time.monotonic() - _T0, 4), "ts": int(time.time() * 1000), "ev": ev}
        rec.update(fields)
        with self._lock:
            if self.keep_in_memory:
                self.events.append(rec)
            if self._fh:
                try:
                    self._fh.write(json.dumps(rec, default=str, separators=(",", ":")) + "\n")
                except Exception:
                    pass
        for fn in self._listeners:
            try:
                fn(rec)
            except Exception:
                pass
        return rec

    def find(self, ev: str) -> List[dict]:
        with self._lock:
            return [e for e in self.events if e["ev"] == ev]


trace = Tracer()


def now_s() -> float:
    """Seconds since process start on the same clock as trace 't'."""
    return time.monotonic() - _T0


def mono_to_t(mono: float) -> float:
    """Convert a time.monotonic() value to the trace clock ('t')."""
    return mono - _T0
