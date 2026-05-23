from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class WakeWordConfig:
    wakeword: str = os.getenv("WAKEWORD", "hey_jarvis")
    sample_rate: int = int(os.getenv("WAKEWORD_SAMPLE_RATE", "16000"))
    chunk_size: int = int(os.getenv("WAKEWORD_CHUNK_SIZE", "1280"))
    threshold: float = float(os.getenv("WAKEWORD_THRESHOLD", "0.5"))
    trigger_level: int = int(os.getenv("WAKEWORD_TRIGGER_LEVEL", "3"))
    vad_threshold: float = float(os.getenv("WAKEWORD_VAD_THRESHOLD", "0.5"))
    enable_vad: bool = os.getenv("WAKEWORD_ENABLE_VAD", "1") == "1"
    enable_speex_noise_suppression: bool = os.getenv("WAKEWORD_ENABLE_SPEEX_NS", "0") == "1"
    microphone_index: Optional[int] = int(os.getenv("WAKEWORD_MIC_INDEX")) if os.getenv("WAKEWORD_MIC_INDEX") else None
    debug_scores: bool = os.getenv("WAKEWORD_DEBUG_SCORES", "1") == "1"
    model_paths: List[str] = field(default_factory=list)


def load_config() -> WakeWordConfig:
    raw_models = os.getenv("WAKEWORD_MODEL_PATHS", "").strip()
    model_paths = [m.strip() for m in raw_models.split(",") if m.strip()]
    cfg = WakeWordConfig(model_paths=model_paths)
    return cfg
