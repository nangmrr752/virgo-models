"""Virgo 1.0 chat model: loads the base model and (if present) the Virgo LoRA adapter."""
import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

SYSTEM = (
    "You are Virgo, an AI assistant made by KSN (Virgo-1.0-Angkor). You are friendly, clear and honest. "
    "Reply in the user's language: Khmer (in Khmer script) when they write Khmer, otherwise English. "
    "Keep answers short unless asked for more. Say so when you are not sure."
)


DEFAULT_BASE = "google/gemma-3-4b-it"
BIG = re.compile(r"(\d+)b", re.I)


def adapter_base(adapter):
    """The base model a Virgo adapter was trained on (from its adapter_config.json)."""
    try:
        with open(os.path.join(adapter, "adapter_config.json"), encoding="utf-8") as f:
            return json.load(f).get("base_model_name_or_path")
    except (OSError, TypeError, ValueError):
        return None


def wants_4bit(base):
    """12B and bigger load in 4 bits on a GPU, so Virgo-1.0-Angkor 12B fits a free 16 GB T4."""
    flag = os.environ.get("VIRGO_4BIT")
    if flag is not None:
        return flag == "1"
    sizes = [int(n) for n in BIG.findall(base.split("/")[-1])]
    return torch.cuda.is_available() and bool(sizes) and max(sizes) >= 12


class VirgoChat:
    def __init__(self, base=None, adapter=None):
        base = base or adapter_base(adapter) or DEFAULT_BASE
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(adapter or base)
        dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if self.device == "cuda" else torch.float32
        kwargs = {"dtype": dtype}
        four_bit = wants_4bit(base)
        if four_bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
            kwargs["device_map"] = {"": 0}
        try:
            model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
        except ValueError:  # Gemma 3 4B and up are image+text models
            from transformers import Gemma3ForConditionalGeneration

            model = Gemma3ForConditionalGeneration.from_pretrained(base, **kwargs)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
        self.model = (model if four_bit else model.to(self.device)).eval()
        self.gemma = "gemma" in base.lower()
        self.base = base
        print(f"Virgo chat: {base}{' (4-bit)' if four_bit else ''}{' + ' + adapter if adapter else ''}")

    def _prompt(self, history, system=SYSTEM):
        # Gemma needs user/assistant turns to alternate: back-to-back turns from the same side (an
        # interrupted or failed answer in realtime voice) are joined, empty ones dropped, and the
        # conversation starts with the user.
        msgs = []
        for m in history:
            content = str(m.get("content") or "").strip()
            if m.get("role") not in ("user", "assistant") or not content:
                continue
            if msgs and msgs[-1]["role"] == m["role"]:
                msgs[-1] = {"role": m["role"], "content": f"{msgs[-1]['content']}\n\n{content}"}
            else:
                msgs.append({"role": m["role"], "content": content})
        while msgs and msgs[0]["role"] != "user":
            msgs.pop(0)
        if self.gemma and msgs:  # Gemma has no system role
            msgs = [{**msgs[0], "content": f"{system}\n\n{msgs[0]['content']}"}] + msgs[1:]
        else:
            msgs = [{"role": "system", "content": system}] + msgs
        # Newer transformers return a dict (input_ids + attention_mask); older ones a tensor.
        out = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        return {k: v.to(self.device) for k, v in out.items()}

    @torch.inference_mode()
    def reply(self, history, max_new_tokens=512, temperature=0.7):
        inputs = self._prompt(history)
        sampling = {"do_sample": True, "temperature": temperature, "top_p": 0.9} if temperature > 0 else {"do_sample": False}
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **sampling)
        return self.tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def stream(self, history, max_new_tokens=512, temperature=0.7):
        """Yields the reply piece by piece (used by the realtime voice so it can speak early)."""
        from threading import Thread

        inputs = self._prompt(history)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        sampling = {"do_sample": True, "temperature": temperature, "top_p": 0.9} if temperature > 0 else {"do_sample": False}
        Thread(target=self.model.generate, kwargs=dict(**inputs, max_new_tokens=max_new_tokens, streamer=streamer, **sampling)).start()
        yield from streamer
