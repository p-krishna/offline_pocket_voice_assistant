import argparse
import io
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional
from wave import open as wave_open

import numpy as np

from common.log import get_logger

log = get_logger("stt")


def samples_to_wav_bytes(samples, sample_rate: int) -> bytes:
    """Accept a flat int16 ndarray or a list of int16 frames → in-memory WAV bytes."""
    if isinstance(samples, np.ndarray):
        flat = samples.astype(np.int16)
    else:
        flat = np.concatenate(samples).astype(np.int16)
    buf = io.BytesIO()
    with wave_open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(flat.tobytes())
    return buf.getvalue()


@dataclass
class STTResult:
    """Transcript plus whisper's own confidence signals (None when the server did not report them)."""
    text: str
    avg_logprob: Optional[float] = None       # mean token log-probability; < about -1.0 is usually garbage
    no_speech_prob: Optional[float] = None    # probability the segment contains no speech at all
    min_word_prob: Optional[float] = None     # least certain word

    def confident(self, min_logprob: float = -0.7) -> bool:
        """False for transcripts whisper itself does not believe. On a labelled sample (120 clips)
        avg_logprob < -0.7 was wrong 8/8 times and never hit a correct transcript."""
        if self.avg_logprob is None:
            return True
        return self.avg_logprob > min_logprob and (self.no_speech_prob or 0.0) < 0.6


class WhisperCppSTT:
    def __init__(self, cfg):
        self.url      = cfg.stt_server_url   # http://127.0.0.1:8081
        self.language = cfg.whisper_language
        self.timeout  = max(30, cfg.http_timeout)
        self.confidence = getattr(cfg, "stt_confidence", True)

    def transcribe(self, samples, sample_rate):
        """Return the transcript ('' if nothing was recognised). Raises on server failure."""
        return self.transcribe_ex(samples, sample_rate).text

    def transcribe_ex(self, samples, sample_rate) -> STTResult:
        """Transcript + confidence. Raises on server failure."""
        if samples is None or (hasattr(samples, "__len__") and len(samples) == 0):
            return STTResult("")

        wav_data = samples_to_wav_bytes(samples, sample_rate)
        boundary = b"----WavBoundary"

        def field(name: str, value: str) -> bytes:
            return (b"--" + boundary + b"\r\n"
                    + f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())

        body = (
            b"--" + boundary + b"\r\n"
            b'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
            b"Content-Type: audio/wav\r\n\r\n" + wav_data + b"\r\n"
            + field("response_format", "verbose_json" if self.confidence else "json")
            + field("language", self.language)
            # Greedy, deterministic decoding: far fewer hallucinated phrases on noise.
            + field("temperature", "0.0")
            + field("temperature_inc", "0.0")
            + b"--" + boundary + b"--\r\n"
        )

        last_err = None
        for attempt in (1, 2):
            req = urllib.request.Request(
                f"{self.url}/inference",
                data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary.decode()}"},
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                return self._parse(result)
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError, ValueError) as e:
                last_err = e
                log.warning("STT request failed (attempt %d/2): %s", attempt, e)
                time.sleep(0.3)
        raise RuntimeError(f"STT server unavailable: {last_err}")


    @staticmethod
    def _parse(result: dict) -> STTResult:
        text = result.get("text", "").strip()
        segs = result.get("segments") or []
        if not segs:
            return STTResult(text)
        lp = [s["avg_logprob"] for s in segs if "avg_logprob" in s]
        nsp = [s["no_speech_prob"] for s in segs if "no_speech_prob" in s]
        words = [w.get("probability") for s in segs for w in (s.get("words") or [])
                 if w.get("probability") is not None and any(c.isalnum() for c in w.get("word", ""))]
        return STTResult(text, sum(lp) / len(lp) if lp else None, max(nsp) if nsp else None,
                         min(words) if words else None)


# --- Standalone test ---
if __name__ == "__main__":
    import sys
    from pathlib import Path
    _SRC = Path(__file__).resolve().parent.parent
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    from common.config import load_config

    parser = argparse.ArgumentParser(description="Test WhisperCppSTT with a WAV file.")
    parser.add_argument("wav_path", type=str, help="Path to the input WAV file.")
    args = parser.parse_args()

    cfg = load_config()
    stt = WhisperCppSTT(cfg)

    with wave_open(args.wav_path, "rb") as wf:
        sample_rate = wf.getframerate()
        frames      = wf.readframes(wf.getnframes())
        samples     = np.frombuffer(frames, dtype=np.int16)

    print("Transcript:", stt.transcribe([samples], sample_rate))
