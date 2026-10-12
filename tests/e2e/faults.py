"""
Fault-injecting proxy in front of the real llama-server.

Lets the E2E suite reproduce the conditions that make a voice assistant feel
broken on slow hardware — without needing slow hardware:

  mode="pass"      forward untouched
  mode="slow"      re-emit tokens at `tokens_per_sec` after `first_token_delay_s`
  mode="stall"     first N tokens normally, then hang forever (stuck decode)
  mode="hang"      accept the request, never answer
  mode="error"     HTTP 500
  mode="refuse"    (use a dead port instead)

It also counts client disconnects, so tests can assert that cancelling a turn
(barge-in) really hangs up on the LLM instead of letting it decode in vain.
"""
from __future__ import annotations

import http.client
import json
import select
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class LLMProxy:
    def __init__(self, port: int = 18080, upstream=("127.0.0.1", 8080)):
        self.port = port
        self.upstream = upstream
        self.mode = "pass"
        self.tokens_per_sec = 6.0
        self.first_token_delay_s = 0.0
        self.stall_after_tokens = 4
        self.requests = 0
        self.disconnects = 0
        self.completed = 0
        self.last_request: dict | None = None
        proxy = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *a):
                pass

            def client_gone(self) -> bool:
                """True once the client hung up (readable socket that yields EOF)."""
                try:
                    r, _, _ = select.select([self.connection], [], [], 0)
                    if r:
                        return self.connection.recv(1, socket.MSG_PEEK) == b""
                except OSError:
                    return True
                return False

            def nap(self, seconds: float) -> None:
                """Sleep, but raise BrokenPipeError as soon as the client disconnects."""
                end = time.monotonic() + seconds
                while time.monotonic() < end:
                    if self.client_gone():
                        raise BrokenPipeError("client gone")
                    time.sleep(min(0.05, max(0.0, end - time.monotonic())))

            def do_GET(self):
                try:
                    c = http.client.HTTPConnection(*proxy.upstream, timeout=5)
                    c.request("GET", self.path)
                    r = c.getresponse()
                    body = r.read()
                    self.send_response(r.status)
                    self.end_headers()
                    self.wfile.write(body)
                except Exception:
                    self.send_response(502)
                    self.end_headers()

            def do_POST(self):
                proxy.requests += 1
                n = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(n)
                try:
                    proxy.last_request = json.loads(body)
                except Exception:
                    pass
                mode = proxy.mode
                if mode == "error":
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(b'{"error":"injected failure"}')
                    return
                if mode == "hang":
                    try:
                        while True:
                            self.nap(0.2)
                    except Exception:
                        proxy.disconnects += 1
                    return
                c = http.client.HTTPConnection(*proxy.upstream, timeout=120)
                c.request("POST", self.path, body=body, headers={"Content-Type": "application/json"})
                r = c.getresponse()
                self.send_response(r.status)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                sent = 0
                try:
                    if mode in ("slow", "stall"):
                        self.nap(proxy.first_token_delay_s)
                    while True:
                        line = r.readline()
                        if not line:
                            break
                        is_delta = line.startswith(b"data:") and b'"content"' in line and b"[DONE]" not in line
                        if mode in ("slow", "stall") and is_delta:
                            if mode == "stall" and sent >= proxy.stall_after_tokens:
                                while True:           # stuck decode
                                    self.nap(0.2)
                            self.nap(1.0 / proxy.tokens_per_sec)
                            sent += 1
                        self.wfile.write(line)
                        self.wfile.flush()
                    proxy.completed += 1
                except (BrokenPipeError, ConnectionResetError, OSError):
                    proxy.disconnects += 1
                finally:
                    try:
                        c.close()
                    except Exception:
                        pass

        self._srv = ThreadingHTTPServer(("127.0.0.1", port), H)
        self._srv.daemon_threads = True
        self._th = threading.Thread(target=self._srv.serve_forever, daemon=True)

    def start(self):
        self._th.start()
        return self

    def stop(self):
        self._srv.shutdown()
        self._srv.server_close()

    def reset(self, mode="pass", **kw):
        self.mode = mode
        self.tokens_per_sec = kw.get("tokens_per_sec", 6.0)
        self.first_token_delay_s = kw.get("first_token_delay_s", 0.0)
        self.stall_after_tokens = kw.get("stall_after_tokens", 4)
        self.requests = self.disconnects = self.completed = 0
