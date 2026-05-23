import numpy as np
import pyaudio
from openwakeword.model import Model
from config import load_config

cfg = load_config()

RATE = cfg.sample_rate
CHUNK = cfg.chunk_size

WAKEWORD = cfg.wakeword
THRESHOLD = cfg.threshold
TRIGGER_LEVEL = cfg.trigger_level

model = Model(
    wakeword_models=[WAKEWORD],
    vad_threshold=cfg.vad_threshold,
    enable_speex_noise_suppression=cfg.enable_speex_noise_suppression,
)

audio = pyaudio.PyAudio()
stream_kwargs = dict(
    format=pyaudio.paInt16,
    channels=1,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK,
)
if cfg.microphone_index is not None:
    stream_kwargs["input_device_index"] = cfg.microphone_index

stream = audio.open(**stream_kwargs)

consecutive_hits = 0

try:
    while True:
        pcm = stream.read(CHUNK, exception_on_overflow=False)
        frame = np.frombuffer(pcm, dtype=np.int16)
        scores = model.predict(frame)

        score = scores.get(WAKEWORD, 0.0)
        
        if score > THRESHOLD:
            consecutive_hits += 1
            if consecutive_hits >= TRIGGER_LEVEL:
                print(f"{WAKEWORD} detected with confidence: {score:.3f}")
                consecutive_hits = 0
        else:
            consecutive_hits = 0

except KeyboardInterrupt:
    pass
finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()