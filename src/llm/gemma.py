import json
import urllib.request

from common.config import load_config


class GemmaLLM:
    def __init__(self, cfg):
        self.url           = cfg.llm_server_url          # http://127.0.0.1:8080
        self.system_prompt = cfg.llm_system_prompt
        self.predict_tokens = cfg.llm_predict_tokens

    def generate(self, transcript):
        if not transcript:
            return ""

        # Build the OpenAI-compatible chat payload.
        payload = json.dumps({
            "messages": [
                {"role": "system",  "content": self.system_prompt},
                {"role": "user",    "content": transcript},
            ],
            "max_tokens": self.predict_tokens,
            "temperature": 0.7,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            timeout=cfg.http_timeout
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"LLM request failed: {e}")
            return "Failed to understand the request. Please try again."

if __name__ == "__main__":
    # Example usage
    cfg = load_config()
    llm = GemmaLLM(cfg)
    response = llm.generate("What can you do?")
    print("LLM Response:", response)