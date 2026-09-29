"""Virgo 1.0 chat model: loads the base model and (if present) the Virgo LoRA adapter."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

SYSTEM = (
    "You are Virgo, an AI assistant made by KSN (Virgo 1.0). You are friendly, clear and honest. "
    "Reply in the user's language: Khmer (in Khmer script) when they write Khmer, otherwise English. "
    "Keep answers short unless asked for more. Say so when you are not sure."
)


class VirgoChat:
    def __init__(self, base="google/gemma-3-4b-it", adapter=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(adapter or base)
        dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if self.device == "cuda" else torch.float32
        try:
            model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype)
        except ValueError:  # Gemma 3 4B and up are image+text models
            from transformers import Gemma3ForConditionalGeneration

            model = Gemma3ForConditionalGeneration.from_pretrained(base, torch_dtype=dtype)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
        self.model = model.to(self.device).eval()
        self.gemma = "gemma" in base.lower()

    def _prompt(self, history, system=SYSTEM):
        msgs = [m for m in history if m["role"] in ("user", "assistant")]
        if self.gemma and msgs:  # Gemma has no system role
            msgs = [{**msgs[0], "content": f"{system}\n\n{msgs[0]['content']}"}] + msgs[1:]
        else:
            msgs = [{"role": "system", "content": system}] + msgs
        return self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self.device)

    @torch.inference_mode()
    def reply(self, history, max_new_tokens=512, temperature=0.7):
        ids = self._prompt(history)
        out = self.model.generate(ids, max_new_tokens=max_new_tokens, do_sample=temperature > 0, temperature=temperature, top_p=0.9)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    def stream(self, history, max_new_tokens=512, temperature=0.7):
        """Yields the reply piece by piece (used by the realtime voice so it can speak early)."""
        from threading import Thread

        ids = self._prompt(history)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        Thread(target=self.model.generate, kwargs=dict(
            input_ids=ids, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
            temperature=temperature, top_p=0.9, streamer=streamer,
        )).start()
        yield from streamer
