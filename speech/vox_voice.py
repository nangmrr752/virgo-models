"""Virgo-1.0-Angkor-Voice: Virgo's voice, built on OpenBMB VoxCPM2 (Apache 2.0, speaks Khmer, English
and 28 more languages). Text → (samples, sample rate).

The voice is a short sample clip, `voice.wav` (made once from a written description with
speech/design_voice.ipynb, so it's Virgo's own voice, not a copy of anyone's), plus an optional LoRA
fine-tune (`lora_weights.safetensors` + `lora_config.json`) that makes it steadier. Both live in the
Hugging Face repo VIRGO_VOX_REPO (e.g. you/Virgo-1.0-Angkor-Voice) or the folder VIRGO_VOX_DIR.

VoxCPM2 needs newer libraries than the Virgo server, so it runs in its own Python environment as a
small worker process the server starts once (same idea as khmer_voice.py):

    VIRGO_VOX_PYTHON=/path/to/vox-env/bin/python   # that environment has `pip install voxcpm`
    VIRGO_VOX_REPO=you/Virgo-1.0-Angkor-Voice
    VIRGO_VOX_DEVICE=cuda:1                         # optional: the second GPU (Kaggle T4 x2)

The worker (`python speech/vox_voice.py --serve <folder>`) reads one JSON line per sentence on stdin
and answers with one JSON line naming a WAV file it wrote.
"""
import json
import os
import subprocess
import sys
import tempfile
import threading

VOICE_DIR = os.environ.get("VIRGO_VOX_DIR", "speech/out/virgo-1.0-angkor-voice")
BASE = os.environ.get("VIRGO_VOX_BASE", "openbmb/VoxCPM2")
FILES = ["voice.wav", "voice.json", "lora_weights.safetensors", "lora_config.json"]
# Used when no voice sample exists yet: a warm, clear, friendly voice described in words.
DEFAULT_DESCRIPTION = "A young woman, warm and friendly, bright clear voice, calm natural pace"


def available():
    """True when the VoxCPM2 environment is set up (the voice sample can come later)."""
    return bool(os.environ.get("VIRGO_VOX_PYTHON"))


class VoxVoice:
    def __init__(self, folder=VOICE_DIR):
        folder = os.path.abspath(folder)
        os.makedirs(folder, exist_ok=True)
        repo = os.environ.get("VIRGO_VOX_REPO")
        if repo and not os.path.exists(os.path.join(folder, "voice.wav")):
            try:
                from huggingface_hub import snapshot_download

                snapshot_download(repo, local_dir=folder, allow_patterns=FILES)
            except Exception as err:  # no trained voice yet: the described voice is used
                print("Virgo voice sample not downloaded, using the described voice:", err)
        python = os.environ.get("VIRGO_VOX_PYTHON", sys.executable)
        self.lock = threading.Lock()
        self.worker = subprocess.Popen([python, os.path.abspath(__file__), "--serve", folder],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8")
        ready = json.loads(self.worker.stdout.readline() or '{"error": "the voice worker stopped"}')
        if "error" in ready:
            raise RuntimeError(f"Virgo voice: {ready['error']}")
        print("Virgo-1.0-Angkor-Voice is ready:", ready)

    def __call__(self, text):
        import soundfile as sf

        with self.lock:
            if self.worker.poll() is not None:
                raise RuntimeError("the voice worker stopped")
            self.worker.stdin.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            self.worker.stdin.flush()
            answer = json.loads(self.worker.stdout.readline() or '{"error": "no answer"}')
        if "error" in answer:
            raise RuntimeError(f"Virgo voice: {answer['error']}")
        try:
            wave, rate = sf.read(answer["path"], dtype="float32")
        finally:
            os.remove(answer["path"])
        return wave, rate


# ---------- The worker (runs inside the VoxCPM2 environment) ----------
def load_model(folder, device=None):
    """VoxCPM2 with Virgo's LoRA when there is one. On GPUs without bfloat16 (T4) it runs in float16."""
    import torch
    from huggingface_hub import snapshot_download
    from voxcpm import VoxCPM

    base = BASE if os.path.isdir(BASE) else snapshot_download(BASE)
    device = device or os.environ.get("VIRGO_VOX_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_bf16_supported():
        config_path = os.path.join(base, "config.json")
        config = json.load(open(config_path))
        if str(config.get("dtype", "")).lower() in ("bfloat16", "bf16"):
            config["dtype"] = "float16"
            json.dump(config, open(config_path, "w"), indent=2)
    lora = folder if os.path.exists(os.path.join(folder, "lora_config.json")) else None
    return VoxCPM.from_pretrained(base, load_denoiser=False, device=device, lora_weights_path=lora)


def voice_settings(folder):
    """What to pass to generate(): the sample clip (and its words) when there is one, else a description."""
    info = {}
    if os.path.exists(os.path.join(folder, "voice.json")):
        info = json.load(open(os.path.join(folder, "voice.json"), encoding="utf-8"))
    sample = os.path.join(folder, "voice.wav")
    if os.path.exists(sample):
        settings = {"reference_wav_path": sample}
        if info.get("text"):  # the sample's exact words: the closest match to the sample's voice
            settings.update(prompt_wav_path=sample, prompt_text=info["text"])
        return settings, ""
    return {}, f"({info.get('description') or DEFAULT_DESCRIPTION})"


def serve(folder):
    import soundfile as sf

    try:
        import inspect
        import random

        import numpy as np
        import torch

        model = load_model(folder)
        accepted = set(inspect.signature(model._generate).parameters)  # older VoxCPM releases have no `seed`
        settings, prefix = voice_settings(folder)
        steps = int(os.environ.get("VIRGO_VOX_STEPS", "8"))  # fewer steps = faster (10 is VoxCPM's default)
        rate = model.tts_model.sample_rate
    except Exception as err:
        print(json.dumps({"error": repr(err)}), flush=True)
        return
    print(json.dumps({"rate": rate, "sample": bool(settings), "steps": steps}), flush=True)
    for line in sys.stdin:
        try:
            text = json.loads(line)["text"]
            random.seed(7); np.random.seed(7); torch.manual_seed(7)  # the same voice every time
            args = {"cfg_value": 2.0, "inference_timesteps": steps, "seed": 7, **settings}
            wave = model.generate(text=prefix + text, **{k: v for k, v in args.items() if k in accepted})
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            sf.write(path, wave, rate)
            print(json.dumps({"path": path}), flush=True)
        except Exception as err:
            print(json.dumps({"error": repr(err)}), flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--serve":
        serve(sys.argv[2])
    else:
        print(__doc__)
