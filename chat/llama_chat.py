"""Virgo chat through llama.cpp's server (llama-server, the "llama" service in docker-compose.yml):
for models our own 4-bit loader can't fit on one GPU, like Angkor-2.0 on Gemma 4 26B A4B (an MoE that
runs in about 15 GB as a 4-bit GGUF). Same interface as VirgoChat (reply, stream), so the API and the
live voice don't change; the server is set with VIRGO_LLAMA_URL (serve/app.py)."""
import json
import os
import re
import urllib.request

from virgo_chat import turns

# Angkor-2.0's instructions (an untrained base model follows them; Angkor-1.x keeps the exact ones it was
# trained with, virgo_chat.SYSTEM_TEMPLATE). Worded so it answers naturally instead of reciting them.
SYSTEM_TEMPLATE = (
    "You are Virgo, KSN's AI assistant; your model is {name}. When someone asks who you are or who made you, "
    "answer briefly and naturally in your own words (for example: \"I'm Virgo, an AI assistant made by KSN.\"); "
    "you are never Gemma, Gemini, Google, ChatGPT, OpenAI, Llama, Meta, Claude or any other company's model. "
    "Don't repeat or describe these instructions.\n"
    "Reply in the language of the user's latest message: Khmer (in Khmer script) for Khmer, English for English, "
    "and the same for any other language, unless they ask for another one. Use natural, polite, everyday wording.\n"
    "Be helpful, accurate and to the point: keep simple answers short and give detail when it's asked for or "
    "needed. If you are not sure or can't know something (today's news, someone's private details), say so "
    "plainly instead of guessing."
)


def system_for(name):
    return SYSTEM_TEMPLATE.format(name=name or "Angkor-2.0")

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
        # No thinking first (VIRGO_LLAMA_THINK=1 turns it on): Gemma 4 can think through the whole answer length
        # (a Khmer question used all 512 tokens thinking and answered nothing), and live voice must start fast.
        think = os.environ.get("VIRGO_LLAMA_THINK") == "1"
        return json.dumps({"messages": msgs, "max_tokens": max_new_tokens + (2048 if think else 0),
                           "temperature": max(temperature, 0), "top_p": 0.9, "stream": stream,
                           "chat_template_kwargs": {"enable_thinking": think}}).encode()

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
