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
    # "onnx" works with NumPy 2 / kokoro-onnx >= 0.5; "tflite" needs NumPy < 2.
    wakeword_framework:                str   = os.getenv("WAKEWORD_FRAMEWORK", "onnx")
    microphone_index:                  Optional[int] = (
        int(os.getenv("MIC_INDEX") or os.getenv("WAKEWORD_MIC_INDEX")) if (os.getenv("MIC_INDEX") or os.getenv("WAKEWORD_MIC_INDEX")) else None
    )
    # Input device name substring (case-insensitive); ignored when MIC_INDEX is set.
    # Set MIC_NAME="" to use the PortAudio default. MIC_REQUIRED=1 aborts if no match.
    microphone_name:                   str   = os.getenv("MIC_NAME", "Jabra")
    microphone_required:               bool  = os.getenv("MIC_REQUIRED", "0") == "1"

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
    utterance_pre_roll_ms:             int   = int(os.getenv("UTTERANCE_PRE_ROLL_MS", "300"))
    # After the wake word fires (late in "Hey Jarvis") keep this much lead-in so STT hears the whole
    # wake phrase (which is then stripped) instead of clipped fragments like "Darvis" / "That's".
    utterance_wake_pre_roll_ms:        int   = int(os.getenv("UTTERANCE_WAKE_PRE_ROLL_MS", "1300"))
    utterance_post_roll_ms:            int   = int(os.getenv("UTTERANCE_POST_ROLL_MS", "300"))
    # Silence after the last detected speech before the utterance is considered
    # finished. ~800 ms feels natural: short enough to answer promptly, long
    # enough not to cut people off while they think mid-sentence.
    utterance_silence_hold_ms:         int   = int(os.getenv("UTTERANCE_SILENCE_HOLD_MS", "800"))
    # Start transcribing speculatively once this much silence followed the speech; the
    # transcript decides how long to keep waiting (see Pipeline._endpoint_hold).
    speculative_stt_ms:                int   = int(os.getenv("SPECULATIVE_STT_MS", "420"))
    # Transcript looks like a finished sentence ("…?" / "…."): answer after this much silence.
    utterance_early_hold_ms:           int   = int(os.getenv("UTTERANCE_EARLY_HOLD_MS", "560"))
    # Transcript trails off (", and the …"): the user is probably still thinking.
    utterance_hesitation_hold_ms:      int   = int(os.getenv("UTTERANCE_HESITATION_HOLD_MS", "1500"))
    # Only the wake phrase so far: wait this long for the actual request before saying "Yes?".
    utterance_wake_hold_ms:            int   = int(os.getenv("UTTERANCE_WAKE_HOLD_MS", "2200"))
    # Follow-up speech must be at least this fraction of the user's usual speaking level.
    followup_level_ratio:              float = float(os.getenv("FOLLOWUP_LEVEL_RATIO", "0.3"))
    # Never end an utterance before this much speech was heard.
    utterance_min_speech_ms:           int   = int(os.getenv("UTTERANCE_MIN_SPEECH_MS", "250"))
    # Hard cap so a noisy room can never record forever.
    utterance_max_ms:                  int   = int(os.getenv("UTTERANCE_MAX_MS", "30000"))
    # Follow-up capture (no wake word) that never reaches min speech within this
    # window is abandoned as a noise trigger.
    utterance_noise_abort_ms:          int   = int(os.getenv("UTTERANCE_NOISE_ABORT_MS", "2500"))

    # Reject tiny, weak captures quietly instead of turning them into STT requests.
    utterance_reject_rms: float = float(os.getenv("UTTERANCE_REJECT_RMS", "0.003"))

    # Conversation-mode re-entry: fresh WebRTC speech frames (30 ms each) required
    # before a follow-up capture starts (no wake word needed while the conversation is open).
    conversation_reentry_start_hits: int = int(os.getenv("CONVERSATION_REENTRY_START_HITS", "5"))
    # ...and the frame RMS must be above this to count (ignores fan hum / hiss).
    conversation_reentry_rms: float = float(os.getenv("CONVERSATION_REENTRY_RMS", "0.006"))

    # Seconds to ignore the mic after any assistant audio finished playing
    # (speaker/room tail). Barge-in detection is unaffected while audio plays.
    assistant_audio_cooldown_s: float = float(os.getenv("ASSISTANT_AUDIO_COOLDOWN_S", "0.25"))

    # --- Slow-inference feedback (keep the user informed while the LLM works) ---
    # Short tick as soon as an utterance is accepted ("I heard you").
    ack_earcon_enabled: bool = os.getenv("ACK_EARCON_ENABLED", "1") == "1"
    # Spoken filler if no answer audio has started this long after the user finished.
    progress_first_phrase_s: float = float(os.getenv("PROGRESS_FIRST_PHRASE_S", "3.5"))
    # Soft "still working" pulse cadence after the first filler phrase.
    progress_pulse_every_s: float = float(os.getenv("PROGRESS_PULSE_EVERY_S", "4.0"))
    # Spoken reassurance cadence for very long waits.
    progress_phrase_every_s: float = float(os.getenv("PROGRESS_PHRASE_EVERY_S", "14.0"))
    # Give up on the LLM if it produced nothing / is stuck for this long.
    llm_first_token_timeout_s: float = float(os.getenv("LLM_FIRST_TOKEN_TIMEOUT_S", "60"))
    llm_stall_timeout_s: float = float(os.getenv("LLM_STALL_TIMEOUT_S", "30"))
    llm_total_timeout_s: float = float(os.getenv("LLM_TOTAL_TIMEOUT_S", "120"))

    # TTS tail: extra seconds of drain after the last sample was written.
    tts_drain_extra_s: float = float(os.getenv("TTS_DRAIN_EXTRA_S", "0.15"))

    # --- Debug ---
    debug_mode:                        bool  = os.getenv("DEBUG_MODE", "1") == "1"
    # Privacy: a voice assistant for people at home should not keep recordings or transcripts by default.
    # Opt in (DEBUG_SAVE_WAV=1 / LOG_TRANSCRIPTS=1) when debugging; saved WAVs are pruned to DEBUG_MAX_WAVS.
    debug_save_wav:                    bool  = os.getenv("DEBUG_SAVE_WAV", "0") == "1"
    debug_max_wavs:                    int   = int(os.getenv("DEBUG_MAX_WAVS", "200"))
    # Without this the trace / turn log / console keep only lengths and timings, not what was said.
    log_transcripts:                   bool  = os.getenv("LOG_TRANSCRIPTS", "0") == "1"
    debug_dir:                         str   = os.getenv("DEBUG_DIR", "debug_audio")
    turnlogdir:                        str   = os.getenv("TURN_LOG_DIR", "debug_logs")
    log_level:                         str   = os.getenv("LOG_LEVEL", "INFO")

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
    # Ask whisper for per-segment confidence (needs serve_whisper.sh --no-language-probabilities to be cheap).
    stt_confidence:                    bool  = os.getenv("STT_CONFIDENCE", "1") == "1"
    # Transcripts with mean token log-probability below this are treated as "didn't understand".
    stt_min_logprob:                   float = float(os.getenv("STT_MIN_LOGPROB", "-0.7"))

    # --- LLM: persistent llama-server ---
    llm_server_url:                    str   = os.getenv("LLM_SERVER_URL", "http://127.0.0.1:8080")
    llm_system_prompt:                 str   = os.getenv(
        "LLM_SYSTEM_PROMPT",
        "You are a helpful German tutor for an English speaker. "
        "Answer clearly, briefly, and in plain spoken German. "
        "Do not use markdown or bullets. "
        "Keep the response focused on teaching. "
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
    http_timeout:                      int   = int(os.environ.get("HTTP_TIMEOUT", "15"))

    # What the assistant says out loud when any stage fails.
    fallback_phrase: str = os.environ.get(
        "FALLBACK_PHRASE", "Sorry, something went wrong. Please try again."
    )

    # --- Conversation memory ---
    memory_turns: int = int(os.environ.get("MEMORY_TURNS", "3"))
    memory_assistant_max_chars: int = int(os.environ.get("MEMORY_ASSISTANT_MAX_CHARS", "240"))

    # --- Echo cancellation (speaker → mic leakage) ---
    aec_enabled: bool = os.environ.get("AEC_ENABLED", "1") == "1"
    # Filter span = partitions * 30 ms; must cover output-device latency + room echo path.
    aec_partitions: int = int(os.environ.get("AEC_PARTITIONS", "14"))
    # Step size relative to total reference power; stable below ~2.5, faster convergence near 2.
    # Play ~2 s of soft noise at startup so the canceller is trained before the first reply.
    aec_calibrate_on_start: bool = os.environ.get("AEC_CALIBRATE_ON_START", "1") == "1"
    aec_mu: float = float(os.environ.get("AEC_MU", "1.6"))
    # Cancellation needed before the barge-in safety margin is dropped. Equal to the convergence threshold
    # (8 dB) on purpose: a stricter value (12 dB) was tried on 2026-10-13 and made real interruptions through
    # echo undetectable, because the measured ERLE is bounded by the room noise floor (it cannot be read as
    # filter quality); the residual-based echo floor in the barge-in detector is the real safeguard.
    aec_trusted_erle_db: float = float(os.environ.get("AEC_TRUSTED_ERLE_DB", "8"))

    # --- Interrupt tuning ---
    interrupt_min_speech_ms: int = int(os.environ.get("INTERRUPT_MIN_SPEECH_MS", "190"))
    interrupt_energy_threshold: float = float(os.environ.get("INTERRUPT_ENERGY_THRESHOLD", "0.005"))
    # Duck our own voice to this gain once barge_in_duck_after_ms of speech was heard
    # (1.0 disables). Gives instant feedback before the stop is confirmed.
    barge_in_duck_gain: float = float(os.environ.get("BARGE_IN_DUCK_GAIN", "0.25"))
    barge_in_duck_after_ms: int = int(os.environ.get("BARGE_IN_DUCK_AFTER_MS", "60"))
    # While we decide whether speech over our answer was a request, a cough or "stop", the answer is paused
    # (not stopped). If nobody resolves the pause within this time it simply continues.
    barge_in_pause_timeout_s: float = float(os.environ.get("BARGE_IN_PAUSE_TIMEOUT_S", "40"))
    interrupt_hiss_filter: bool = os.environ.get("INTERRUPT_HISS_FILTER", "1") == "1"
    # Barge-in energy must exceed (estimated speaker->mic echo) * this factor.
    interrupt_echo_margin: float = float(os.environ.get("INTERRUPT_ECHO_MARGIN", "2.0"))

    # --- Conversation timeout ---
    conversation_timeout_s: int = int(os.environ.get("CONVERSATION_TIMEOUT_S", "45"))
    conversation_warning_s: int = int(os.environ.get("CONVERSATION_WARNING_S", "8"))

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
