from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class Config:
    # --- Wake word ---
    wakeword:                          str   = os.getenv("WAKEWORD", "hey_jarvis")
    sample_rate:                       int   = int(os.getenv("WAKEWORD_SAMPLE_RATE", "16000"))
    wakeword_chunk_size:               int   = int(os.getenv("WAKEWORD_CHUNK_SIZE", "480"))
    wakeword_threshold:                float = float(os.getenv("WAKEWORD_THRESHOLD", "0.5"))
    wakeword_trigger_level:            int   = int(os.getenv("WAKEWORD_TRIGGER_LEVEL", "3"))
    wakeword_vad_threshold:            float = float(os.getenv("WAKEWORD_VAD_THRESHOLD", "0.5"))
    wakeword_enable_speex_noise_suppression: bool = os.getenv("WAKEWORD_ENABLE_SPEEX_NS", "0") == "1"
    microphone_index:                  Optional[int] = (
        int(os.getenv("WAKEWORD_MIC_INDEX")) if os.getenv("WAKEWORD_MIC_INDEX") else None
    )

    model_paths:                       List[str] = field(default_factory=list)

    # --- WebRTC VAD ---
    webrtc_frame_ms:                   int   = int(os.getenv("WEBRTC_FRAME_MS", "30"))
    webrtc_aggressiveness:             int   = int(os.getenv("WEBRTC_AGGRESSIVENESS", "2"))
    webrtc_start_hits:                 int   = int(os.getenv("WEBRTC_START_HITS", "2"))
    webrtc_stop_hits:                  int   = int(os.getenv("WEBRTC_STOP_HITS", "3"))

    # --- Silero VAD ---
    silero_chunk_size:                 int   = int(os.getenv("SILERO_CHUNK_SIZE", "512"))
    silero_threshold:                  float = float(os.getenv("SILERO_THRESHOLD", "0.5"))
    silero_history:                    int   = int(os.getenv("SILERO_HISTORY", "5"))
    silero_stop_silence_frames:        int   = int(os.getenv("SILERO_STOP_SILENCE_FRAMES", "10"))

    # --- Utterance timing ---
    utterance_pre_roll_ms:             int   = int(os.getenv("UTTERANCE_PRE_ROLL_MS", "250"))
    utterance_post_roll_ms:            int   = int(os.getenv("UTTERANCE_POST_ROLL_MS", "500"))
    utterance_silence_hold_ms:         int   = int(os.getenv("UTTERANCE_SILENCE_HOLD_MS", "2000"))
    utterance_min_ms:                  int   = int(os.getenv("UTTERANCE_MIN_MS", "2000"))

    # Minimum ms before any early finalization is allowed.
    utterance_floor_ms: int = int(os.getenv("UTTERANCE_FLOOR_MS", "700"))

    # Silero avg probability below this means very confident silence → allow early exit.
    silero_early_exit_threshold: float = float(os.getenv("SILERO_EARLY_EXIT_THRESHOLD", "0.05"))

    # Reject tiny, weak captures quietly instead of turning them into STT requests.
    utterance_reject_ms: int = int(os.getenv("UTTERANCE_REJECT_MS", "500"))
    utterance_reject_rms: float = float(os.getenv("UTTERANCE_REJECT_RMS", "0.003"))

    # Conversation-mode re-entry: require this many fresh WebRTC speech frames
    # before starting a new capture (prevents immediate re-triggers after TTS).
    conversation_reentry_start_hits: int = int(os.getenv("CONVERSATION_REENTRY_START_HITS", "5"))
    # Sustained WebRTC speech frames required before re-entering conversation.
    # Higher = harder to accidentally re-trigger after TTS. Tune with noisy env logs.

    # Seconds to ignore mic after any assistant/system audio.
    assistant_audio_cooldown_s: float = float(os.getenv("ASSISTANT_AUDIO_COOLDOWN_S", "2.0"))
    # Seconds to ignore mic after any assistant/system audio.
    # 2.0s gives speaker hardware time to drain + avoids self-hearing TTS tail.

    # Extra seconds to sleep after writing all PCM samples to PyAudio before
    # calling stop_stream(). This drains the hardware output buffer so the
    # last ~0.2-0.5 s of each TTS sentence is not clipped.
    # Set to 0.0 to disable (restores old clipping behaviour).
    tts_drain_extra_s: float = float(os.getenv("TTS_DRAIN_EXTRA_S", "0.15"))

    tts_playback_guard_s: float = float(os.getenv("TTS_PLAYBACK_GUARD_S", "0.5"))
    # Extra cooldown applied at the START of TTS playback (during speaking).
    # Prevents wake word false triggers caused by the opening of the TTS phrase.
    # This stacks with assistant_audio_cooldown_s (applied at end of TTS).

    # --- Debug ---
    debug_mode:                        bool  = os.getenv("DEBUG_MODE", "1") == "1"
    debug_save_wav:                    bool  = os.getenv("DEBUG_SAVE_WAV", "1") == "1"
    debug_dir:                         str   = os.getenv("DEBUG_DIR", "debug_audio")
    turnlogdir:                        str   = os.getenv("TURN_LOG_DIR", "debug_logs")

    # --- STT: persistent whisper-server ---
    stt_server_url:                    str   = os.getenv("STT_SERVER_URL", "http://127.0.0.1:8081")
    whisper_bin:                       str   = os.getenv(
        "WHISPER_BIN",
        "/home/puli/projects/whisper/whisper.cpp/build/bin/whisper-server",
    )
    whisper_model:                     str   = os.getenv(
        "WHISPER_MODEL",
        "/home/puli/projects/whisper/whisper.cpp/models/ggml-tiny.en.bin",
    )
    whisper_language:                  str   = os.getenv("WHISPER_LANGUAGE", "en")

    # --- LLM: persistent llama-server ---
    llm_server_url:                    str   = os.getenv("LLM_SERVER_URL", "http://127.0.0.1:8080")
    llm_system_prompt:                 str   = os.getenv(
        "LLM_SYSTEM_PROMPT",
        "You are a helpful offline voice assistant for a visually impaired user. "
        "Answer clearly, briefly, and in plain spoken English. "
        "Do not use markdown or bullets. "
        "Keep the response focused on the question. "
        "Be polite, concise and positive.",
    )
    llm_predict_tokens:                int   = int(os.getenv("LLM_PREDICT_TOKENS", "150"))

    # --- TTS: persistent kokoro server ---
    tts_server_url:                    str   = os.getenv("TTS_SERVER_URL", "http://127.0.0.1:8082")
    tts_server_port:                   int   = int(os.getenv("TTS_SERVER_PORT", "8082"))
    kokoro_model_path:                 str   = os.getenv(
        "KOKORO_MODEL_PATH", "/home/puli/projects/kokoro/kokoro-v1.0.onnx"
    )
    kokoro_voices_path:                str   = os.getenv(
        "KOKORO_VOICES_PATH", "/home/puli/projects/kokoro/voices.json"
    )
    tts_voice:                         str   = os.getenv("TTS_VOICE", "af")
    tts_output_dir:                    str   = os.getenv("TTS_OUTPUT_DIR", "debug_audio")

    # Seconds to wait for any server HTTP response before giving up.
    http_timeout:                      int   = int(os.environ.get("HTTP_TIMEOUT", "10"))

    # What the assistant says out loud when any stage fails.
    fallback_phrase: str = os.environ.get(
        "FALLBACK_PHRASE", "Sorry, something went wrong. Please try again."
    )

    # --- Conversation memory ---
    memory_turns: int = int(os.environ.get("MEMORY_TURNS", "2"))
    memory_assistant_max_chars: int = int(os.environ.get("MEMORY_ASSISTANT_MAX_CHARS", "100"))

    # --- Interrupt tuning ---
    interrupt_min_speech_ms: int = int(os.environ.get("INTERRUPT_MIN_SPEECH_MS", "300"))
    interrupt_energy_threshold: float = float(os.environ.get("INTERRUPT_ENERGY_THRESHOLD", "0.005"))
    interrupt_hiss_filter: bool = os.environ.get("INTERRUPT_HISS_FILTER", "1") == "1"

    # --- Conversation timeout ---
    conversation_timeout_s: int = int(os.environ.get("CONVERSATION_TIMEOUT_S", "45"))
    conversation_warning_s: int = int(os.environ.get("CONVERSATION_WARNING_S", "5"))

    # --- Voice selection ---
    system_voice: str = os.environ.get("SYSTEM_VOICE", "af")
    response_voice: str = os.environ.get("RESPONSE_VOICE", "af")

    # Enable per-turn latency and counter logging to terminal.
    metrics_enabled: bool = os.getenv("METRICS_ENABLED", "1") == "1"

    false_wake_log_interval: int = int(os.getenv("FALSE_WAKE_LOG_INTERVAL", "10"))
    # Log a false-wake summary every N false wake events.
    # Set to 0 to disable interval logging (still logged at exit).

    # Print a cumulative summary when the pipeline exits cleanly.
    metrics_exit_summary: bool = os.getenv("METRICS_EXIT_SUMMARY", "1") == "1"


def load_config() -> Config:
    raw_models = os.getenv("WAKEWORD_MODEL_PATHS", "").strip()
    model_paths = [m.strip() for m in raw_models.split(",") if m.strip()]
    return Config(model_paths=model_paths)
