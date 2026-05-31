import signal

import pyaudio
import webrtcvad

from common import now
from common.config import load_config


cfg = load_config()


class WebRTCGate:
    def __init__(self):
        self.sample_rate = cfg.sample_rate
        self.frame_ms = cfg.webrtc_frame_ms
        self.frame_samples = int(self.sample_rate * self.frame_ms / 1000)
        self.vad = webrtcvad.Vad(cfg.webrtc_aggressiveness)
        self.mic_index = cfg.microphone_index
        self.state = None
        self.started_at = None
        self.running = True

    def stop(self, *_):
        self.running = False

    def open(self):
        audio = pyaudio.PyAudio()
        kwargs = dict(format=pyaudio.paInt16, channels=1, rate=self.sample_rate, input=True, frames_per_buffer=self.frame_samples)
        if self.mic_index is not None:
            kwargs['input_device_index'] = self.mic_index
        return audio, audio.open(**kwargs)

    def run(self, on_change):
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        audio, stream = self.open()
        try:
            while self.running:
                pcm = stream.read(self.frame_samples, exception_on_overflow=False)
                new_state = 'speech' if self.vad.is_speech(pcm, self.sample_rate) else 'silence'
                t = now()
                if self.state is None:
                    self.state = new_state
                    self.started_at = t
                    on_change(new_state, None, self.started_at, t)
                    continue
                if new_state != self.state:
                    old = self.state
                    started = self.started_at
                    self.state = new_state
                    self.started_at = t
                    on_change(new_state, old, started, t)
        finally:
            stream.stop_stream(); stream.close(); audio.terminate()
