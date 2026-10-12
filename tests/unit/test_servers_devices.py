"""Server table derives from config, port checks, and microphone device resolution."""
import socket

import pytest

import common.servers as servers
from audio import devices


def test_servers_follow_config_urls(monkeypatch):
    monkeypatch.setenv("LLM_SERVER_URL", "http://127.0.0.1:9100")
    import importlib
    import common.config as config
    importlib.reload(config)
    try:
        s = servers._build_servers()
        llm = [x for x in s if x["name"] == "LLM"][0]
        assert llm["port"] == 9100
        assert llm["url"] == "http://127.0.0.1:9100/health"
    finally:
        monkeypatch.delenv("LLM_SERVER_URL")
        importlib.reload(config)


def test_port_in_use_detects_a_bound_but_silent_server():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    port = sock.getsockname()[1]
    try:
        assert servers.port_in_use({"host": "127.0.0.1", "port": port})
    finally:
        sock.close()
    assert not servers.port_in_use({"host": "127.0.0.1", "port": port})


class FakePA:
    def __init__(self, devs, default):
        self.devs, self.default = devs, default

    def get_device_count(self):
        return len(self.devs)

    def get_device_info_by_index(self, i):
        d = self.devs[i]
        return {"name": d[0], "maxInputChannels": d[1], "defaultSampleRate": 16000.0, "index": i}

    def get_default_input_device_info(self):
        return {"index": self.default}


PA = FakePA([("Jabra BIZ 1500: USB Audio (hw:1,0)", 1), ("HDA Intel PCH: ALC (hw:0,0)", 2), ("HDMI out", 0)], 1)


def test_name_beats_default():
    assert devices.resolve_input_device(PA, None, "jabra").index == 0


def test_index_beats_name():
    assert devices.resolve_input_device(PA, 1, "jabra").index == 1


def test_unmatched_name_falls_back_to_default_unless_required():
    assert devices.resolve_input_device(PA, None, "nonexistent").index == 1
    with pytest.raises(devices.MicError):
        devices.resolve_input_device(PA, None, "nonexistent", require=True)


def test_output_only_device_is_not_an_input():
    with pytest.raises(devices.MicError):
        devices.resolve_input_device(PA, 2, None)
