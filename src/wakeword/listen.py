import argparse
import signal

import numpy as np
import pyaudio
from openwakeword.model import Model

from common import now
from common.config import load_config


cfg = load_config()
parser = argparse.ArgumentParser(description='Wake word listener')
parser.add_argument('--list-devices', action='store_true', help='List audio devices and exit')
args = parser.parse_args()

if args.list_devices:
    audio = pyaudio.PyAudio()
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        print(f"[{i}] {info.get('name')} | input={int(info.get('maxInputChannels', 0))} | output={int(info.get('maxOutputChannels', 0))}")
    audio.terminate()
    raise SystemExit(0)


class WakeWordListener:
    def __init__(self):
        self.wakeword = cfg.wakeword
        self.sample_rate = cfg.sample_rate
        self.chunk_size = cfg.wakeword_chunk_size
        self.threshold = cfg.wakeword_threshold
        self.trigger_level = cfg.wakeword_trigger_level
        self.model = Model(wakeword_models=[self.wakeword], vad_threshold=cfg.wakeword_vad_threshold, enable_speex_noise_suppression=cfg.wakeword_enable_speex_noise_suppression)
        self.mic_index = cfg.microphone_index
        self.hits = 0
        self.running = True

    def stop(self, *_):
        self.running = False

    def open(self):
        audio = pyaudio.PyAudio()
        kwargs = dict(format=pyaudio.paInt16, channels=1, rate=self.sample_rate, input=True, frames_per_buffer=self.chunk_size)
        if self.mic_index is not None:
            kwargs['input_device_index'] = self.mic_index
        return audio, audio.open(**kwargs)

    def run(self, on_detect, on_score=None):
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        audio, stream = self.open()
        try:
            while self.running:
                pcm = stream.read(self.chunk_size, exception_on_overflow=False)
                frame = np.frombuffer(pcm, dtype=np.int16)
                score = self.model.predict(frame).get(self.wakeword, 0.0)
                if on_score:
                    on_score(score)
                if score >= self.threshold:
                    self.hits += 1
                else:
                    self.hits = 0
                if self.hits >= self.trigger_level:
                    self.hits = 0
                    on_detect(score, now())
        finally:
            stream.stop_stream(); stream.close(); audio.terminate()
