import json
import urllib.request

from common.config import load_config


class GemmaLLM:
    def __init__(self, cfg):
        self.url            = cfg.llm_server_url          # http://127.0.0.1:8080
        self.system_prompt  = cfg.llm_system_prompt
        self.predict_tokens = cfg.llm_predict_tokens
        self.timeout        = cfg.http_timeout

    def generate(self, transcript, history=None):
        # history is a flat list of {"role": "user"/"assistant", "content": "..."}
        # pairs from previous turns, oldest first.
        if not transcript:
            return ""

        # Build messages: system → history turns → current user message.
        # In OpenAI API format, but compatible with any LLM server that uses the same structure.
        messages = [{"role": "system", "content": self.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": transcript})

        payload = json.dumps({
            "messages": messages,
            "max_tokens": self.predict_tokens,
            "temperature": 0.7,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[LLM] Request failed: {e}")
            raise


# --- Standalone test ---
if __name__ == "__main__":
    import sys
    from pathlib import Path
    _SRC = Path(__file__).resolve().parent.parent
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    cfg = load_config()
    llm = GemmaLLM(cfg)
    response = llm.generate("What can you do?")
    print("LLM Response:", response)