import subprocess
import time
import urllib.request
import urllib.error

# ── Server definitions ────────────────────────────────────────────────────────
# Each entry describes one server:
#   name    — label for log messages
#   port    — TCP port to probe
#   url     — HTTP URL to ping (a lightweight endpoint)
#   cmd     — shell command to start the server (same as Makefile)
#   env     — extra env vars needed at launch (e.g. PYTHONPATH)
SERVERS = [
    {
        "name": "LLM",
        "port": 8080,
        "url":  "http://127.0.0.1:8080/health",
        "cmd":  "/bin/bash src/llm/serve_gemma.sh",
        "env":  {},
    },
    {
        "name": "STT",
        "port": 8081,
        "url":  "http://127.0.0.1:8081/health",
        "cmd":  "/bin/bash src/stt/serve_whisper.sh",
        "env":  {},
    },
    {
        "name": "TTS",
        "port": 8082,
        "url":  "http://127.0.0.1:8082/health",
        "cmd":  "python src/tts/serve_kokoro.py",
        "env":  {"PYTHONPATH": "src"},
    },
]

# ── Single-server helpers ─────────────────────────────────────────────────────

def _ping(url: str, timeout: int = 2) -> bool:
    """Return True if the server responds to an HTTP GET on its health URL."""
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except Exception:
        return False


def _start(server: dict) -> subprocess.Popen:
    """
    Launch a server as a background subprocess.
    Returns the Popen handle so the caller can track it.
    """
    import os
    env = {**os.environ, **server["env"]}
    proc = subprocess.Popen(
        server["cmd"],
        shell=True,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"[servers] {server['name']} started (pid={proc.pid})")
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
    Raises RuntimeError if any server never responds within max_wait seconds.
    """
    deadline = time.monotonic() + max_wait

    for srv in SERVERS:
        print(f"[servers] Waiting for {srv['name']} at {srv['url']} ...", flush=True)
        while time.monotonic() < deadline:
            if _ping(srv["url"]):
                print(f"[servers] {srv['name']} ready ✓")
                break
            time.sleep(poll_interval)
        else:
            raise RuntimeError(
                f"[servers] {srv['name']} did not respond within {max_wait}s. "
                "Is the server process running?"
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
                    if proc is not None and proc.poll() is None:
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