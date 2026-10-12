import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse
import urllib.request
import urllib.error

_ROOT = Path(__file__).resolve().parents[2]
_LOG_DIR = Path(os.getenv("SERVER_LOG_DIR", str(_ROOT / "debug_logs")))


def _build_servers() -> list:
    """
    The three servers, with URLs/ports taken from the configuration (LLM_SERVER_URL, STT_SERVER_URL,
    TTS_SERVER_URL) so that health checks and recovery always look where the clients actually talk.
      name — label for log messages        url — health endpoint
      cmd  — shell command that starts it  env — extra environment at launch
    """
    from common.config import load_config
    cfg = load_config()

    def spec(name, base_url, cmd, env):
        u = urlparse(base_url)
        return {"name": name, "port": u.port or 80, "host": u.hostname or "127.0.0.1",
                "url": base_url.rstrip("/") + "/health", "cmd": cmd, "env": env}

    return [
        spec("LLM", cfg.llm_server_url, "/bin/bash src/llm/serve_gemma.sh", {}),
        spec("STT", cfg.stt_server_url, "/bin/bash src/stt/serve_whisper.sh", {}),
        spec("TTS", cfg.tts_server_url, f"{sys.executable} -u src/tts/serve_kokoro.py", {"PYTHONPATH": "src"}),
    ]


SERVERS = _build_servers()

# ── Single-server helpers ─────────────────────────────────────────────────────

def _ping(url: str, timeout: int = 2) -> bool:
    """Return True if the server responds to an HTTP GET on its health URL."""
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except Exception:
        return False


def port_in_use(server: dict, timeout: float = 0.5) -> bool:
    """
    True when something accepts TCP connections on the server's port. A server that is loading its
    model (or hung) is not healthy, but starting a second copy would only fight over the port and the
    memory — so callers must wait for it instead of spawning.
    """
    try:
        with socket.create_connection((server.get("host", "127.0.0.1"), server["port"]), timeout=timeout):
            return True
    except OSError:
        return False


def _start(server: dict) -> subprocess.Popen:
    """
    Launch a server as a background subprocess; returns the Popen handle.
    Output goes to debug_logs/server_<name>.log (it used to be discarded, which made a server that
    failed to start undiagnosable).
    """
    env = {**os.environ, **server["env"]}
    log_path = _LOG_DIR / f"server_{server['name'].lower()}.log"
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        out = open(log_path, "ab")
    except OSError:
        out, log_path = subprocess.DEVNULL, None
    proc = subprocess.Popen(
        server["cmd"],
        shell=True,
        cwd=str(_ROOT),              # commands use project-relative paths
        env=env,
        stdout=out,
        stderr=subprocess.STDOUT if log_path else subprocess.DEVNULL,
        start_new_session=True,      # survive Ctrl+C / exit of whoever restarted them
    )
    print(f"[servers] {server['name']} started (pid={proc.pid})" + (f", log: {log_path}" if log_path else ""))
    return proc


# ── Startup: wait until all servers are ready ────────────────────────────────

def wait_for_servers(
    max_wait: int = 120,   # give up after this many seconds total
    poll_interval: float = 2.0,
) -> None:
    """
    Ping each server in a retry loop until it responds or max_wait is reached.
    Replaces the fixed `sleep 5` in the Makefile.

    Called once at pipeline startup — blocks until all three servers are up.
    Raises RuntimeError if any server never responds within max_wait seconds (each server gets its own).
    """
    for srv in SERVERS:
        deadline = time.monotonic() + max_wait          # per server: the LLM load must not eat the others' time
        print(f"[servers] Waiting for {srv['name']} at {srv['url']} ...", flush=True)
        while time.monotonic() < deadline:
            if _ping(srv["url"]):
                print(f"[servers] {srv['name']} ready ✓")
                break
            time.sleep(poll_interval)
        else:
            raise RuntimeError(
                f"[servers] {srv['name']} did not respond within {max_wait}s. "
                f"Is the server process running? (output: {_LOG_DIR / ('server_' + srv['name'].lower() + '.log')})"
            )


# ── Watchdog: restart any server that goes down ───────────────────────────────

def watchdog(poll_interval: float = 10.0) -> None:
    """
    Background-safe loop that checks all three servers every poll_interval seconds
    and restarts any that have gone down.

    Run this as a standalone script:
        PYTHONPATH=src python src/common/servers.py

    Or call it from a dedicated make target (see Makefile).
    The loop runs forever; Ctrl+C stops it cleanly.
    """
    # Track live subprocess handles so we can check if they died.
    procs: dict[str, subprocess.Popen | None] = {s["name"]: None for s in SERVERS}

    print("[watchdog] Started — polling every", poll_interval, "seconds.")
    try:
        while True:
            for srv in SERVERS:
                name = srv["name"]
                alive = _ping(srv["url"])

                if not alive:
                    # Check if a previously-started proc already exited.
                    proc = procs[name]
                    if port_in_use(srv):
                        # Bound but not healthy: loading its model, busy, or started by someone else.
                        print(f"[watchdog] {name} port {srv['port']} is bound but not healthy, waiting...")
                    elif proc is not None and proc.poll() is None:
                        # Process is still running but not responding yet — give it time.
                        print(f"[watchdog] {name} not responding, waiting...")
                    else:
                        print(f"[watchdog] {name} is down — restarting...")
                        procs[name] = _start(srv)

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\n[watchdog] Stopped.")


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    watchdog()