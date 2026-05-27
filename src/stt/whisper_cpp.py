import io
import json
import tempfile
import argparse
import urllib.request
from pathlib import Path
from wave import open as wave_open

import numpy as np


class WhisperCppSTT:
    def __init__(self, cfg):
        self.url      = cfg.stt_server_url   # http://127.0.0.1:8081
        self.language = cfg.whisper_language

    def transcribe(self, samples, sample_rate):
        if not samples:
            return ""

        # Write samples to a temp WAV file — whisper-server needs a real file.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with wave_open(str(tmp_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(np.concatenate(samples).astype(np.int16).tobytes())

            # Send WAV as multipart form to the persistent whisper-server.
            with open(tmp_path, "rb") as f:
                wav_data = f.read()

            boundary = b"----WavBoundary"
            body = (
                b"--" + boundary + b"\r\n"
                b'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
                b"Content-Type: audio/wav\r\n\r\n"
                + wav_data + b"\r\n"
                b"--" + boundary + b"\r\n"
                b'Content-Disposition: form-data; name="response_format"\r\n\r\n'
                b"json\r\n"
                b"--" + boundary + b"--\r\n"
            )

            req = urllib.request.Request(
                f"{self.url}/inference",
                data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary.decode()}"},
                timeout=cfg.http_timeout
            )

            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            # whisper-server returns { "text": "..." } in JSON mode.
            return result.get("text", "").strip()

        except Exception as e:
            print(f"[STT] Request failed: {e}")
            return ""

        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass


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
