import signal

import numpy as np
import pyaudio
import torch
from silero_vad import load_silero_vad

from common import now
from common.config import load_config

torch.set_num_threads(1)
cfg = load_config()


class SileroGate:
    def __init__(self):
        self.sample_rate = cfg.sample_rate
        self.chunk_size = cfg.silero_chunk_size
        self.threshold = cfg.silero_threshold
        self.mic_index = cfg.microphone_index
        self.model = load_silero_vad()
        self.history = []
        self.history_max = cfg.silero_history
        self.state = None
        self.started_at = None
        self.running = True

    def stop(self, *_):
        self.running = False

    def open(self):
        audio = pyaudio.PyAudio()
        kwargs = dict(format=pyaudio.paInt16, channels=1, rate=self.sample_rate, input=True, frames_per_buffer=self.chunk_size)
        if self.mic_index is not None:
            kwargs['input_device_index'] = self.mic_index
        return audio, audio.open(**kwargs)

    def predict(self, pcm):
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        x = torch.from_numpy(x)
        return self.model(x, self.sample_rate).item()

    def update(self, prob):
        self.history.append(prob)
        if len(self.history) > self.history_max:
            self.history.pop(0)
        avg = sum(self.history) / len(self.history)
        new_state = 'speech' if avg >= self.threshold else 'silence'
        t = now()
        old = self.state
        changed = new_state != self.state
        if self.state is None:
            self.state = new_state
            self.started_at = t
            return True, new_state, None, avg, self.started_at, t
        if changed:
            started = self.started_at
            self.state = new_state
            self.started_at = t
            return True, new_state, old, avg, started, t
        return False, new_state, old, avg, self.started_at, t
