"""
LLM client for the local llama-server (OpenAI-compatible API).

`stream()` yields text deltas as they are generated so the pipeline can start
speaking the first sentence while the rest is still being decoded — the single
biggest latency win on a 5–15 tok/s model. It is cancellable (barge-in) and has
separate first-token / stall / total timeouts instead of one blunt request
timeout, so a slow-but-alive model is never mistaken for a dead one.
"""
import http.client
import json
import queue
import socket
import threading
import time
from typing import Callable, Iterator, Optional
from urllib.parse import urlparse

from common.config import load_config
from common.log import get_logger

log = get_logger("llm")


class LLMError(RuntimeError):
    """Server unreachable / returned an error."""


class LLMTimeout(LLMError):
    """No token (first-token) or no progress (stall) within the allowed time."""


class _SSEReader(threading.Thread):
    """Reads the streaming HTTP response on its own thread so the consumer can
    enforce timeouts and cancel by shutting the socket down."""

    def __init__(self, url: str, payload: bytes, connect_timeout: float):
        super().__init__(name="LLMReader", daemon=True)
        u = urlparse(url)
        self.host, self.port = u.hostname, u.port or 80
        self.payload = payload
        self.connect_timeout = connect_timeout
        self.q: "queue.Queue[tuple]" = queue.Queue()
        self._conn: Optional[http.client.HTTPConnection] = None
        # http.client forgets conn.sock once a close-delimited response begins, so keep our own
        # handle: it is the only way to hang up while the reader thread is blocked in recv().
        self._sock: Optional[socket.socket] = None
        self._closed = False

    def run(self) -> None:
        try:
            self._conn = http.client.HTTPConnection(self.host, self.port, timeout=self.connect_timeout)
            self._conn.connect()
            self._sock = self._conn.sock
            # The connect timeout must not apply while we wait for the first token:
            # a slow model may send no response headers for many seconds. The consumer
            # (GemmaLLM.stream) enforces first-token / stall / total timeouts itself.
            self._conn.sock.settimeout(None)
            self._conn.request("POST", "/v1/chat/completions", body=self.payload,
                               headers={"Content-Type": "application/json"})
            resp = self._conn.getresponse()
            if resp.status != 200:
                body = resp.read(500).decode("utf-8", "replace")
                self.q.put(("error", LLMError(f"HTTP {resp.status}: {body}")))
                return
            while True:
                raw = resp.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    choice = json.loads(data)["choices"][0]
                except (ValueError, KeyError, IndexError):
                    continue
                delta = (choice.get("delta") or {}).get("content")
                if delta:
                    self.q.put(("delta", delta))
                if choice.get("finish_reason"):
                    self.q.put(("finish", choice["finish_reason"]))
            self.q.put(("done", None))
        except Exception as e:
            if not self._closed:
                self.q.put(("error", LLMError(f"{type(e).__name__}: {e}")))
            else:
                self.q.put(("done", None))
        finally:
            self.abort()

    def abort(self) -> None:
        """Hang up. llama-server stops generating once the client disconnects."""
        self._closed = True
        sock = self._sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        conn = self._conn
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


class GemmaLLM:
    def __init__(self, cfg):
        self.url = cfg.llm_server_url
        self.system_prompt = cfg.llm_system_prompt
        self.predict_tokens = cfg.llm_predict_tokens
        self.first_token_timeout = cfg.llm_first_token_timeout_s
        self.stall_timeout = cfg.llm_stall_timeout_s
        self.total_timeout = cfg.llm_total_timeout_s
        self.last_finish_reason: Optional[str] = None
        self.last_first_token_s: Optional[float] = None
        self.last_tokens = 0

    def _messages(self, transcript, history):
        messages = [{"role": "system", "content": self.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": transcript})
        return messages

    def stream(self, transcript: str, history=None,
               cancel: Callable[[], bool] = lambda: False) -> Iterator[str]:
        """Yield text deltas. Returns quietly if cancelled. Raises LLMError/LLMTimeout."""
        if not transcript:
            return
        payload = json.dumps({
            "messages": self._messages(transcript, history),
            "max_tokens": self.predict_tokens,
            "temperature": 0.7,
            "stream": True,
            "cache_prompt": True,
        }).encode("utf-8")

        reader = _SSEReader(self.url, payload, connect_timeout=5.0)
        t0 = time.monotonic()
        last_progress = t0
        got_first = False
        self.last_finish_reason = None
        self.last_first_token_s = None
        self.last_tokens = 0
        reader.start()
        try:
            while True:
                if cancel():
                    log.info("LLM stream cancelled by caller after %.1fs", time.monotonic() - t0)
                    return
                try:
                    kind, val = reader.q.get(timeout=0.1)
                except queue.Empty:
                    now = time.monotonic()
                    if now - t0 > self.total_timeout:
                        raise LLMTimeout(f"total generation time exceeded {self.total_timeout:.0f}s")
                    limit = self.stall_timeout if got_first else self.first_token_timeout
                    if now - last_progress > limit:
                        what = "stalled mid-answer" if got_first else "no first token"
                        raise LLMTimeout(f"{what} for {limit:.0f}s")
                    continue
                if kind == "delta":
                    last_progress = time.monotonic()
                    self.last_tokens += 1
                    if not got_first:
                        got_first = True
                        self.last_first_token_s = last_progress - t0
                    yield val
                elif kind == "finish":
                    self.last_finish_reason = val
                elif kind == "done":
                    return
                elif kind == "error":
                    raise val
        finally:
            reader.abort()

    def warm_up(self) -> None:
        """Prime llama-server's KV cache with the system prompt so the first real question
        does not pay for evaluating it. Best-effort, never raises."""
        try:
            t0 = time.monotonic()
            for _ in self.stream("Hello.", None):
                break
            log.info("LLM warm-up done in %.1fs", time.monotonic() - t0)
        except Exception as e:
            log.debug("LLM warm-up skipped: %s", e)

    def generate(self, transcript, history=None) -> str:
        """Blocking convenience wrapper (used by tools / standalone check)."""
        return "".join(self.stream(transcript, history)).strip()


# --- Standalone test ---
if __name__ == "__main__":
    import sys
    from pathlib import Path
    _SRC = Path(__file__).resolve().parent.parent
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    cfg = load_config()
    llm = GemmaLLM(cfg)
    t0 = time.monotonic()
    for piece in llm.stream("What can you do?"):
        print(f"[{time.monotonic() - t0:5.2f}s] {piece!r}")
    print("finish:", llm.last_finish_reason, "first token:", llm.last_first_token_s)
