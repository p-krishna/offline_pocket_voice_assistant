"""Audio I/O. Importing this package also silences ALSA's (harmless but very noisy) plugin probing."""
import ctypes as _ct


def _silence_alsa() -> None:
    try:
        err_t = _ct.CFUNCTYPE(None, _ct.c_char_p, _ct.c_int, _ct.c_char_p, _ct.c_int, _ct.c_char_p)
        handler = err_t(lambda *args: None)
        _ct.cdll.LoadLibrary("libasound.so.2").snd_lib_error_set_handler(handler)
        globals()["_alsa_handler"] = handler          # keep a reference alive
    except Exception:
        pass


_silence_alsa()

from audio.speaker import AudioItem, Speaker, SpeakerBackend, PyAudioBackend  # noqa: F401
from audio.mic import MicBase, PyAudioMic  # noqa: F401
from audio import sounds  # noqa: F401
