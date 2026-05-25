from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class Config:
    # --- Wake word ---
    wakeword:                          str   = os.getenv('WAKEWORD', 'hey_jarvis')
    sample_rate:                       int   = int(os.getenv('WAKEWORD_SAMPLE_RATE', '16000'))
    wakeword_chunk_size:               int   = int(os.getenv('WAKEWORD_CHUNK_SIZE', '480'))
    wakeword_threshold:                float = float(os.getenv('WAKEWORD_THRESHOLD', '0.5'))
    wakeword_trigger_level:            int   = int(os.getenv('WAKEWORD_TRIGGER_LEVEL', '3'))
    wakeword_vad_threshold:            float = float(os.getenv('WAKEWORD_VAD_THRESHOLD', '0.5'))
    wakeword_enable_speex_noise_suppression: bool = os.getenv('WAKEWORD_ENABLE_SPEEX_NS', '0') == '1'
    microphone_index:                  Optional[int] = (
        int(os.getenv('WAKEWORD_MIC_INDEX')) if os.getenv('WAKEWORD_MIC_INDEX') else None
    )
    model_paths:                       List[str] = field(default_factory=list)

    # --- WebRTC VAD ---
    webrtc_frame_ms:                   int   = int(os.getenv('WEBRTC_FRAME_MS', '30'))
    webrtc_aggressiveness:             int   = int(os.getenv('WEBRTC_AGGRESSIVENESS', '2'))
    webrtc_start_hits:                 int   = int(os.getenv('WEBRTC_START_HITS', '2'))
    webrtc_stop_hits:                  int   = int(os.getenv('WEBRTC_STOP_HITS', '3'))

    # --- Silero VAD ---
    silero_chunk_size:                 int   = int(os.getenv('SILERO_CHUNK_SIZE', '512'))
    silero_threshold:                  float = float(os.getenv('SILERO_THRESHOLD', '0.5'))
    silero_history:                    int   = int(os.getenv('SILERO_HISTORY', '5'))
    silero_stop_silence_frames:        int   = int(os.getenv('SILERO_STOP_SILENCE_FRAMES', '10'))

    # --- Utterance timing ---
    utterance_pre_roll_ms:             int   = int(os.getenv('UTTERANCE_PRE_ROLL_MS', '250'))
    utterance_post_roll_ms:            int   = int(os.getenv('UTTERANCE_POST_ROLL_MS', '500'))
    utterance_silence_hold_ms:         int   = int(os.getenv('UTTERANCE_SILENCE_HOLD_MS', '2000'))
    utterance_min_ms:                  int   = int(os.getenv('UTTERANCE_MIN_MS', '2000'))

    # --- Debug ---
    debug_mode:                        bool  = os.getenv('DEBUG_MODE', '1') == '1'
    debug_save_wav:                    bool  = os.getenv('DEBUG_SAVE_WAV', '1') == '1'
    debug_dir:                         str   = os.getenv('DEBUG_DIR', 'debug_audio')

    # --- STT: persistent whisper-server ---
    stt_server_url:                    str   = os.getenv('STT_SERVER_URL', 'http://127.0.0.1:8081')
    whisper_bin:                       str   = os.getenv(
        'WHISPER_BIN',
        '/home/puli/projects/whisper/whisper.cpp/build/bin/whisper-server'
    )
    whisper_model:                     str   = os.getenv(
        'WHISPER_MODEL',
        '/home/puli/projects/whisper/whisper.cpp/models/ggml-tiny.en.bin'
    )
    whisper_language:                  str   = os.getenv('WHISPER_LANGUAGE', 'en')

    # --- LLM: persistent llama-server ---
    llm_server_url:                    str   = os.getenv('LLM_SERVER_URL', 'http://127.0.0.1:8080')
    llm_system_prompt:                 str   = os.getenv(
        'LLM_SYSTEM_PROMPT',
        'You are a helpful offline voice assistant for a visually impaired user. '
        'Answer clearly, briefly, and in plain spoken English. '
        'Do not use markdown or bullets. '
        'Keep the response focused on the question. '
        'Be polite, concise and positive.'
    )
    llm_predict_tokens:                int   = int(os.getenv('LLM_PREDICT_TOKENS', '150'))

    # --- TTS: persistent kokoro server ---
    tts_server_url:                    str   = os.getenv('TTS_SERVER_URL',   'http://127.0.0.1:8082')
    tts_server_port:                   int   = int(os.getenv('TTS_SERVER_PORT', '8082'))
    kokoro_model_path:                 str   = os.getenv(
        'KOKORO_MODEL_PATH', '/home/puli/projects/kokoro/kokoro-v1.0.onnx'
    )
    kokoro_voices_path:                str   = os.getenv(
        'KOKORO_VOICES_PATH', '/home/puli/projects/kokoro/voices.json'
    )
    tts_voice:                         str   = os.getenv('TTS_VOICE',      'af')
    tts_output_dir:                    str   = os.getenv('TTS_OUTPUT_DIR', 'debug_audio')


def load_config() -> Config:
    raw_models  = os.getenv('WAKEWORD_MODEL_PATHS', '').strip()
    model_paths = [m.strip() for m in raw_models.split(',') if m.strip()]
    return Config(model_paths=model_paths)
