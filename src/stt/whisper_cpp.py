import subprocess
import tempfile
import argparse
from pathlib import Path
from wave import open as wave_open
import numpy as np
from common.config import load_config

class WhisperCppSTT:
    def __init__(self, cfg, test_file=None):
        self.bin      = cfg.whisper_bin
        self.model    = cfg.whisper_model
        self.language = cfg.whisper_language
        self.test_file = test_file

    def transcribe(self, samples, sample_rate):
        if not samples:
            return ""
        # Write audio to a temp WAV, pass to whisper-cli, then delete the file.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            with wave_open(str(tmp_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(np.concatenate(samples).astype(np.int16).tobytes())
            cmd = [self.bin, "--no-prints", "--no-timestamps", "-m", self.model, "-f", str(tmp_path)]
            if self.language:
                cmd += ["-l", self.language]
            result = subprocess.run(cmd, capture_output=True, text=True)
            output = (result.stdout or "") + "\n" + (result.stderr or "")
            lines = []
            for line in output.splitlines():
                s = line.strip()
                if not s:
                    continue
                # Skip whisper internal log lines.
                if s.startswith(("system_info", "main :", "whisper_model_load",
                                  "whisper_print_timings", "load", "ggml_", "[")):
                    continue
                if s.startswith("***") or s.startswith("error:"):
                    continue
                lines.append(s)
            return " ".join(lines).strip()
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Test WhisperCppSTT with a WAV file.")
    parser.add_argument("wav_path", type=str, help="Path to the input WAV file.")
    args = parser.parse_args()
    cfg = load_config()

    stt = WhisperCppSTT(cfg, test_file=args.wav_path)
    with wave_open(args.wav_path, "rb") as wf:
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16)
    print(stt.transcribe([samples], sample_rate))