"""
World = one fully wired assistant (real models, real servers) living in a simulated room.
"""
from __future__ import annotations

import logging
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common.config import load_config            # noqa: E402
from common.log import mono_to_t, trace         # noqa: E402
from tests.e2e.sim import Room, SimBackend, SimMic, UserSim, SR  # noqa: E402

ASSISTANT_PROMPT = (
    "You are a helpful offline voice assistant. Answer clearly, briefly, and in plain spoken English, "
    "in at most three short sentences. Do not use markdown, lists or emoji. You are offline: you cannot browse the "
    "internet, set reminders or take actions; if you do not know something, say so in one short sentence."
)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    severity: str = "fail"      # fail | warn

    def __str__(self):
        mark = "PASS" if self.ok else ("WARN" if self.severity == "warn" else "FAIL")
        return f"[{mark}] {self.name}" + (f" — {self.detail}" if self.detail else "")


class _ErrorCounter(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records: List[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


class World:
    def __init__(self, name: str, out_dir: Path, *, noise_rms=0.002, echo_gain=0.0,
                 cfg_overrides: Optional[dict] = None, llm_url: Optional[str] = None,
                 user_voice="am_adam"):
        from pipeline.assistant_pipeline import Pipeline
        self.name = name
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = load_config()
        self.cfg.llm_system_prompt = ASSISTANT_PROMPT
        self.cfg.debug_save_wav = False
        self.cfg.turnlogdir = str(out_dir / "turns")
        self.cfg.tts_output_dir = str(ROOT / "debug_audio")
        if llm_url:
            self.cfg.llm_server_url = llm_url
        for k, v in (cfg_overrides or {}).items():
            setattr(self.cfg, k, v)
        self.room = Room(noise_rms=noise_rms, echo_gain=echo_gain)
        self.user = UserSim(self.room, default_voice=user_voice)
        self.mic = SimMic(self.room)
        self.backend = SimBackend(self.room)
        trace.open(str(out_dir / "trace.jsonl"))
        self._errs = _ErrorCounter()
        logging.getLogger("assistant").addHandler(self._errs)
        self.pipe = Pipeline(cfg=self.cfg, mic=self.mic, speaker_backend=self.backend)
        self.t_start = None

    # ── lifecycle ─────────────────────────────────────────────────────────
    def start(self):
        self.pipe.start(wait_servers=False)
        self.pipe.speaker.wait_idle(10)
        self.t_start = time.monotonic()
        time.sleep(0.3)

    def stop(self):
        self.pipe.stop()
        logging.getLogger("assistant").removeHandler(self._errs)
        self.save_audio()

    @property
    def log_problems(self) -> List[logging.LogRecord]:
        return self._errs.records

    # ── trace access ──────────────────────────────────────────────────────
    @property
    def events(self) -> List[dict]:
        with trace._lock:
            return list(trace.events)

    def mark(self) -> int:
        return len(self.events)

    def find(self, ev: str, since: int = 0, **match) -> List[dict]:
        out = []
        for e in self.events[since:]:
            if e["ev"] == ev and all(e.get(k) == v for k, v in match.items()):
                out.append(e)
        return out

    def wait_for(self, pred: Callable[[], bool], timeout: float, poll: float = 0.05) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if pred():
                return True
            time.sleep(poll)
        return pred()

    def wait_ev(self, ev: str, since: int = 0, timeout: float = 30, n: int = 1, **match) -> Optional[dict]:
        ok = self.wait_for(lambda: len(self.find(ev, since, **match)) >= n, timeout)
        got = self.find(ev, since, **match)
        return got[n - 1] if ok else None

    def wait_quiet(self, timeout: float = 90, settle: float = 0.8) -> bool:
        """Wait until no turn is active, nothing is playing and `settle` seconds passed since."""
        def quiet():
            return (not self.pipe.turn_active.is_set() and not self.pipe.speaker.is_active()
                    and self.pipe.utterance_queue.empty())
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if quiet():
                time.sleep(settle)
                if quiet():
                    return True
            time.sleep(0.05)
        return False

    # ── user behaviours ───────────────────────────────────────────────────
    def wake_and_say(self, request: str, *, pause: float = 0.45, **kw):
        """'Hey Jarvis' … (short natural pause) … request — what most people do."""
        if pause <= 0:
            return self.user.say(f"Hey Jarvis, {request}", **kw)
        return self.user.say_parts([("Hey Jarvis.", pause), (request, 0.0)], **kw)

    # ── artifacts ─────────────────────────────────────────────────────────
    def save_audio(self):
        import json
        gt = [{"kind": "user", "text": d["text"], "t0": round(self.tt(d["t_start"]), 3), "t1": round(self.tt(d["t_end"]), 3)}
              for d in self.user.said]
        gt += [{"kind": c.kind, "text": c.label, "t0": round(self.tt(c.t_start), 3), "t1": round(self.tt(c.t_end), 3)}
               for c in self.room.all_clips if c.kind != "user"]
        (self.out_dir / "ground_truth.json").write_text(json.dumps(sorted(gt, key=lambda x: x["t0"]), indent=1))
        if not self.room.mic_log:
            return
        mic = np.concatenate(self.room.mic_log)
        spk = np.zeros_like(mic)
        t0 = self.mic._start
        for ts, p in self.backend.record:
            a = int((ts - t0) * SR)
            if a < 0 or a >= len(spk):
                continue
            seg = p[: len(spk) - a]
            spk[a:a + len(seg)] += seg
        sf.write(self.out_dir / "mic.wav", mic, SR)
        sf.write(self.out_dir / "assistant.wav", np.clip(spk, -1, 1), SR)
        st = np.stack([mic, np.clip(spk, -1, 1)], axis=1)
        sf.write(self.out_dir / "stereo_mic_L_assistant_R.wav", st, SR)

    def tt(self, mono: float) -> float:
        """monotonic → trace time"""
        return mono_to_t(mono)
