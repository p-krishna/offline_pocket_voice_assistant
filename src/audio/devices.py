"""Explicit input-device selection. Never trust the OS/PortAudio default blindly.

Why: the conda ALSA has no PulseAudio plugin, so PortAudio's "default" device
resolves to the onboard card 0 even when PulseAudio's default source is the USB
headset. We therefore pick the input by index (MIC_INDEX) or name (MIC_NAME,
default "Jabra") and log exactly what was chosen and why.

    python -m audio.devices          # list input devices and the selection
    python -m audio.devices --test   # also record 3 s and print the level
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional

import pyaudio

from common.log import get_logger

log = get_logger("mic")


class MicError(RuntimeError):
    pass


@dataclass
class InputDevice:
    index: int
    name: str
    channels: int
    default_rate: float
    is_default: bool = False

    def __str__(self) -> str:
        return (f"[{self.index}] {self.name} (in={self.channels}ch, "
                f"native {self.default_rate:.0f} Hz{', PortAudio default' if self.is_default else ''})")


def list_input_devices(pa: pyaudio.PyAudio) -> List[InputDevice]:
    try:
        default_idx = pa.get_default_input_device_info()["index"]
    except Exception:
        default_idx = None
    out = []
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if int(info.get("maxInputChannels", 0)) > 0:
            out.append(InputDevice(i, info["name"], int(info["maxInputChannels"]),
                                   float(info["defaultSampleRate"]), i == default_idx))
    return out


def resolve_input_device(pa: pyaudio.PyAudio, index: Optional[int] = None,
                         name: Optional[str] = None, require: bool = False) -> InputDevice:
    """index > name substring > PortAudio default (with a warning). Raises MicError if
    an explicit choice cannot be satisfied, or if `require` and the name is not found."""
    devices = list_input_devices(pa)
    log.info("input devices found: %s", "; ".join(map(str, devices)) or "NONE")
    if not devices:
        raise MicError("no audio input devices found (is the headset plugged in?)")
    if index is not None:
        for d in devices:
            if d.index == index:
                log.info("selected input %s — reason: MIC_INDEX=%d", d, index)
                return d
        raise MicError(f"MIC_INDEX={index} is not an input device; available: {[str(d) for d in devices]}")
    if name:
        needle = name.lower()
        for d in devices:
            if needle in d.name.lower():
                log.info("selected input %s — reason: name matches MIC_NAME=%r", d, name)
                return d
        msg = f"no input device matching MIC_NAME={name!r}; available: {[str(d) for d in devices]}"
        if require:
            raise MicError(msg)
        log.warning(msg)
    d = next((d for d in devices if d.is_default), devices[0])
    log.warning("falling back to PortAudio default input %s — this may NOT be your headset; "
                "set MIC_NAME or MIC_INDEX", d)
    return d


def resolve_from_cfg(pa: pyaudio.PyAudio, cfg) -> InputDevice:
    return resolve_input_device(pa, cfg.microphone_index, cfg.microphone_name,
                                require=cfg.microphone_required)


def main() -> None:
    from common.config import load_config
    cfg = load_config()
    pa = pyaudio.PyAudio()
    try:
        for d in list_input_devices(pa):
            print(d)
        dev = resolve_from_cfg(pa, cfg)
        print("SELECTED:", dev)
        if "--test" in sys.argv:
            import numpy as np
            s = pa.open(format=pyaudio.paInt16, channels=1, rate=cfg.sample_rate, input=True,
                        input_device_index=dev.index, frames_per_buffer=480)
            print("Speak now (3 s)...", flush=True)
            for k in range(3):
                d = np.frombuffer(b"".join(s.read(1600, exception_on_overflow=False) for _ in range(10)), np.int16)
                print(f"  second {k+1}: rms={np.sqrt((d.astype(float)**2).mean()):7.1f} peak={abs(d).max()}")
            s.close()
    finally:
        pa.terminate()


if __name__ == "__main__":
    main()
