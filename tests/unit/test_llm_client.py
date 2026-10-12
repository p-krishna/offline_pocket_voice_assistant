"""LLM streaming client against a local fake llama-server: streaming, cancel, timeouts, errors."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from llm.gemma import GemmaLLM, LLMError, LLMTimeout


class Fake(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, mode="ok", delay=0.0):
        self.mode, self.delay = mode, delay
        self.disconnected = threading.Event()
        super().__init__(("127.0.0.1", 0), _H)


class _H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self.rfile.read(n)
        srv = self.server
        if srv.mode == "http500":
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"boom")
            return
        time.sleep(srv.delay)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def ev(obj):
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
            self.wfile.flush()

        try:
            if srv.mode == "ok":
                for w in ["Hello", " there", "."]:
                    ev({"choices": [{"delta": {"content": w}}]})
                ev({"choices": [{"delta": {}, "finish_reason": "stop"}]})
                self.wfile.write(b"data: [DONE]\n\n")
            elif srv.mode == "endless":
                while True:
                    ev({"choices": [{"delta": {"content": "x "}}]})
                    time.sleep(0.05)
            elif srv.mode == "stall":
                ev({"choices": [{"delta": {"content": "one"}}]})
                time.sleep(30)
        except (BrokenPipeError, ConnectionResetError, OSError):
            srv.disconnected.set()


def make(mode, delay=0.0, **cfg):
    srv = Fake(mode, delay)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    c = dict(llm_server_url=f"http://127.0.0.1:{srv.server_address[1]}", llm_system_prompt="sys",
             llm_predict_tokens=20, llm_first_token_timeout_s=5, llm_stall_timeout_s=5, llm_total_timeout_s=20)
    c.update(cfg)
    return srv, GemmaLLM(SimpleNamespace(**c))


def test_streams_deltas_and_finish_reason():
    srv, llm = make("ok")
    assert "".join(llm.stream("hi")) == "Hello there."
    assert llm.last_finish_reason == "stop"
    assert llm.last_first_token_s is not None
    srv.shutdown()


def test_http_error_raises_llm_error():
    srv, llm = make("http500")
    with pytest.raises(LLMError):
        list(llm.stream("hi"))
    srv.shutdown()


def test_first_token_timeout_is_a_timeout_not_a_dead_server():
    srv, llm = make("ok", delay=3.0, llm_first_token_timeout_s=0.5)
    with pytest.raises(LLMTimeout):
        list(llm.stream("hi"))
    srv.shutdown()


def test_slow_first_token_within_limit_is_fine():
    srv, llm = make("ok", delay=1.0, llm_first_token_timeout_s=5)
    assert "".join(llm.stream("hi")) == "Hello there."
    srv.shutdown()


def test_stall_mid_answer_times_out():
    srv, llm = make("stall", llm_stall_timeout_s=0.5)
    got = []
    with pytest.raises(LLMTimeout):
        for d in llm.stream("hi"):
            got.append(d)
    assert got == ["one"]
    srv.shutdown()


def test_cancel_hangs_up_on_the_server():
    srv, llm = make("endless")
    n = 0
    t0 = time.monotonic()
    for _ in llm.stream("hi", cancel=lambda: n >= 3):
        n += 1
    assert time.monotonic() - t0 < 3
    assert srv.disconnected.wait(3), "cancelling must close the connection so the server stops decoding"
    srv.shutdown()
