from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class ListenerConfig:
    wakeword: str = os.getenv('WAKEWORD', 'hey_jarvis')
    sample_rate: int = int(os.getenv('WAKEWORD_SAMPLE_RATE', '16000'))
    wakeword_chunk_size: int = int(os.getenv('WAKEWORD_CHUNK_SIZE', '480'))
    wakeword_threshold: float = float(os.getenv('WAKEWORD_THRESHOLD', '0.5'))
    wakeword_trigger_level: int = int(os.getenv('WAKEWORD_TRIGGER_LEVEL', '3'))
    wakeword_vad_threshold: float = float(os.getenv('WAKEWORD_VAD_THRESHOLD', '0.5'))
    wakeword_enable_speex_noise_suppression: bool = os.getenv('WAKEWORD_ENABLE_SPEEX_NS', '0') == '1'
    microphone_index: Optional[int] = int(os.getenv('WAKEWORD_MIC_INDEX')) if os.getenv('WAKEWORD_MIC_INDEX') else None

    webrtc_frame_ms: int = int(os.getenv('WEBRTC_FRAME_MS', '30'))
    webrtc_aggressiveness: int = int(os.getenv('WEBRTC_AGGRESSIVENESS', '2'))
    webrtc_start_hits: int = int(os.getenv('WEBRTC_START_HITS', '2'))
    webrtc_stop_hits: int = int(os.getenv('WEBRTC_STOP_HITS', '3'))

    silero_chunk_size: int = int(os.getenv('SILERO_CHUNK_SIZE', '512'))
    silero_threshold: float = float(os.getenv('SILERO_THRESHOLD', '0.5'))
    silero_history: int = int(os.getenv('SILERO_HISTORY', '5'))
    silero_stop_silence_frames: int = int(os.getenv('SILERO_STOP_SILENCE_FRAMES', '10'))

    utterance_pre_roll_ms: int = int(os.getenv('UTTERANCE_PRE_ROLL_MS', '250'))
    utterance_post_roll_ms: int = int(os.getenv('UTTERANCE_POST_ROLL_MS', '500'))
    utterance_silence_hold_ms: int = int(os.getenv('UTTERANCE_SILENCE_HOLD_MS', '2000'))
    utterance_min_ms: int = int(os.getenv('UTTERANCE_MIN_MS', '2000'))

    model_paths: List[str] = field(default_factory=list)

    debug_mode: bool = os.getenv('DEBUG_MODE', '1') == '1'
    debug_save_wav: bool = os.getenv('DEBUG_SAVE_WAV', '1') == '1'
    debug_dir: str = os.getenv('DEBUG_DIR', 'debug_audio')


def load_config() -> ListenerConfig:
    raw_models = os.getenv('WAKEWORD_MODEL_PATHS', '').strip()
    model_paths = [m.strip() for m in raw_models.split(',') if m.strip()]
    return ListenerConfig(model_paths=model_paths)
