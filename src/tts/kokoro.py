import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import pyaudio
from kokoro_onnx import Kokoro
from common.config import load_config


class KokoroTTS:
    def __init__(self, cfg):
        # Load Kokoro model once; reuse for every synthesis call.
        self.tts         = Kokoro(model_path=cfg.kokoro_model_path, voices_path=cfg.kokoro_voices_path)
        self.voice       = cfg.tts_voice
        self.sample_rate = cfg.tts_playback_rate
        self.out_dir     = Path(cfg.tts_output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def clean(self, text):
        # Strip markdown so spoken output sounds natural.
        text = re.sub(r"\*\*(.+?)\*\*",  r"\1", text)   # bold
        text = re.sub(r"\*(.+?)\*",      r"\1", text)   # italic
        text = re.sub(r"`(.+?)`",        r"\1", text)   # code
        text = re.sub(r"^#{1,6}\s+",     "",    text, flags=re.M)  # headings
        text = re.sub(r"^[-•*]\s+",      "",    text, flags=re.M)  # bullets
        text = re.sub(r"\s+",            " ",   text)   # collapse whitespace
        return text.strip()

    def synthesize(self, text):
        # Clean the text, then generate audio samples as a numpy float32 array.
        clean_text = self.clean(text)
        if not clean_text:
            return None, ""
        audio, sample_rate = self.tts.create(clean_text, voice=self.voice, speed=1.0)
        return audio, sample_rate, clean_text

    def save(self, audio, sample_rate):
        # Save the generated audio as a timestamped WAV file.
        path = self.out_dir / f"tts_{int(time.time())}.wav"
        sf.write(str(path), audio, sample_rate)
        return path

    def play(self, audio, sample_rate):
        # Convert float32 audio to int16 PCM and play via PyAudio.
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        pa  = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            output=True,
        )
        stream.write(pcm)
        stream.stop_stream()
        stream.close()
        pa.terminate()

    def speak(self, text):
        # Full pipeline: clean -> synthesize -> save -> play.
        audio, sample_rate, clean_text = self.synthesize(text)
        if audio is None:
            print("TTS: nothing to speak.")
            return
        path = self.save(audio, sample_rate)
        print(f"TTS saved : {path}")
        self.play(audio, sample_rate)
        print(f"TTS played: {clean_text[:60]}...")


# --- Standalone test: run this file directly to verify TTS works ---
if __name__ == "__main__":
    cfg = load_config()
    tts = KokoroTTS(cfg=cfg)
    test_text = (
        "Hello! This is a test of your offline Kokoro voice assistant. "
        "If you can hear this clearly, the text to speech stage is working correctly."
    )
    print("Testing TTS...")
    tts.speak(test_text)
    print("Done.")