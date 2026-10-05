"""Virgo chat through llama.cpp's server (llama-server, the "llama" service in docker-compose.yml):
for models our own 4-bit loader can't fit on one GPU, like Angkor-2.0 on Gemma 4 26B A4B (an MoE that
runs in about 15 GB as a 4-bit GGUF). Same interface as VirgoChat (reply, stream), so the API and the
live voice don't change; the server is set with VIRGO_LLAMA_URL (serve/app.py)."""
import json
import re
import urllib.request

from virgo_chat import system_for, turns

# Thinking that comes inside the text (reasoning models): only the answer is kept.
THINKING = re.compile(r"<think>.*?</think>|<\|channel\|>analysis.*?<\|end\|>", re.S)


class LlamaChat:
    def __init__(self, url, name="Angkor-2.0", timeout=600):
        self.url = url.rstrip("/")
        self.name = name
        self.system = system_for(name)
        self.timeout = timeout
        print(f"Virgo chat: {name} through llama.cpp at {self.url}")

    def _body(self, history, max_new_tokens, temperature, system=None, stream=False):
        msgs = [{"role": "system", "content": system or self.system}] + turns(history)
        return json.dumps({"messages": msgs, "max_tokens": max_new_tokens, "temperature": max(temperature, 0),
                           "top_p": 0.9, "stream": stream}).encode()

    def _request(self, body):
        return urllib.request.Request(self.url + "/v1/chat/completions", data=body, method="POST",
                                      headers={"content-type": "application/json"})

    def reply(self, history, max_new_tokens=512, temperature=0.7):
        with urllib.request.urlopen(self._request(self._body(history, max_new_tokens, temperature)), timeout=self.timeout) as res:
            message = json.loads(res.read())["choices"][0]["message"]
        return THINKING.sub("", message.get("content") or "").strip()

    def stream(self, history, max_new_tokens=512, temperature=0.7, system=None):
        """Yields the reply piece by piece (server-sent events from llama-server); thinking is left out."""
        req = self._request(self._body(history, max_new_tokens, temperature, system, stream=True))
        thinking = False
        with urllib.request.urlopen(req, timeout=self.timeout) as res:
            for raw in res:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0].get("delta", {})
                except (ValueError, KeyError, IndexError):
                    continue
                piece = delta.get("content") or ""  # reasoning_content (the thinking) is skipped
                if "<think>" in piece:
                    thinking, piece = True, piece.split("<think>")[0]
                if thinking:
                    if "</think>" not in piece:
                        continue
                    thinking, piece = False, piece.split("</think>", 1)[1]
                if piece:
                    yield piece
