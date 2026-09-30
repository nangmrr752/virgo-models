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
import re
import threading

VOICE_DIR = os.environ.get("VIRGO_VOX_DIR", "speech/out/virgo-1.0-angkor-voice")
BASE = os.environ.get("VIRGO_VOX_BASE", "openbmb/VoxCPM2")
FILES = ["voice.wav", "voice.json", "voice_en.wav", "voice_en.json", "lora_weights.safetensors", "lora_config.json"]
# Used when no voice sample exists yet: a warm, clear, friendly voice described in words.
KHMER = re.compile(r"[\u1780-\u17ff\u19e0-\u19ff]")
# Languages VoxCPM2 speaks; the others use Kokoro-82M (Apache 2.0), a separate, much faster English
# voice. VIRGO_VOX_LANGS=km,en makes VoxCPM2 speak English too (with voice_en.wav when there is one).
VOX_LANGS = set(os.environ.get("VIRGO_VOX_LANGS", "km").split(","))
ENGLISH_VOICE = os.environ.get("VIRGO_ENGLISH_VOICE", "af_heart")  # a warm, clear Kokoro voice
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
        # The worker's messages (and errors) go to a log file, so a crash says why.
        self.log_path = os.path.join(tempfile.gettempdir(), "virgo-voice-worker.log")
        log = open(self.log_path, "w")
        self.worker = subprocess.Popen([python, os.path.abspath(__file__), "--serve", folder],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, encoding="utf-8")
        ready = json.loads(self.worker.stdout.readline() or "{}")
        if "error" in ready or not ready:
            raise RuntimeError(f"Virgo voice: {ready.get('error') or self.why_stopped()}")
        print("Virgo-1.0-Angkor-Voice is ready:", ready)

    def why_stopped(self):
        """Why the worker ended: killed for memory, or the last lines it printed."""
        try:
            code = self.worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            code = None
        tail = "".join(open(self.log_path, errors="replace").readlines()[-15:]).strip()
        if code in (-9, 137):
            return f"the voice worker was killed (exit {code}): out of memory (RAM). Last lines:\n{tail}"
        return f"the voice worker stopped (exit {code}). Last lines:\n{tail}"

    def __call__(self, text, lang=None):
        """lang: "km" or "en" (worked out from the text when not given)."""
        import soundfile as sf

        lang = lang or ("km" if KHMER.search(text) else "en")
        with self.lock:
            if self.worker.poll() is not None:
                raise RuntimeError(f"Virgo voice: {self.why_stopped()}")
            self.worker.stdin.write(json.dumps({"text": text, "lang": lang}, ensure_ascii=False) + "\n")
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
    # bfloat16 needs an Ampere GPU (compute 8.0+); a T4 (7.5) only emulates it, very slowly. Newer torch
    # reports emulated support as "supported", so check the GPU itself. VoxCPM2's config.json has no
    # dtype when it's the bfloat16 default, so it's written out. VIRGO_VOX_DTYPE overrides.
    dtype = os.environ.get("VIRGO_VOX_DTYPE")
    if not dtype and device.startswith("cuda"):
        index = int(device.split(":")[1]) if ":" in device else 0
        dtype = "float16" if torch.cuda.get_device_capability(index)[0] < 8 else None
    if dtype:
        config_path = os.path.join(base, "config.json")
        config = json.load(open(config_path))
        if config.get("dtype") != dtype:
            config["dtype"] = dtype
            text = json.dumps(config, indent=2)
            os.remove(config_path)  # a link into the download cache: replace it, don't write through it
            open(config_path, "w").write(text)
    lora = folder if os.path.exists(os.path.join(folder, "lora_config.json")) else None
    kwargs = {"load_denoiser": False, "device": device, "lora_weights_path": lora}
    # VIRGO_VOX_OPTIMIZE=0 skips torch.compile: less memory while loading (Colab's 12 GB), a bit slower.
    import inspect

    if os.environ.get("VIRGO_VOX_OPTIMIZE") == "0" and "optimize" in inspect.signature(VoxCPM.from_pretrained).parameters:
        kwargs["optimize"] = False
    # VoxCPM2 builds its 2B model in float32 before converting it: about 8 GB. Building it straight on
    # the GPU keeps that out of RAM (Colab has 12 GB; the worker was killed for memory there).
    if device.startswith("cuda"):
        with torch.device(device):
            return VoxCPM.from_pretrained(base, **kwargs)
    return VoxCPM.from_pretrained(base, **kwargs)


def voice_settings(folder, lang="km"):
    """What to pass to generate(): the sample clip (and its words) when there is one, else a description.
    English uses voice_en.wav when it exists (a sample spoken in English sounds most natural)."""
    name = "voice_en" if lang == "en" and os.path.exists(os.path.join(folder, "voice_en.wav")) else "voice"
    info = {}
    if os.path.exists(os.path.join(folder, f"{name}.json")):
        info = json.load(open(os.path.join(folder, f"{name}.json"), encoding="utf-8"))
    sample = os.path.join(folder, f"{name}.wav")
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
        voices = {lang: voice_settings(folder, lang) for lang in ("km", "en")}
        steps = int(os.environ.get("VIRGO_VOX_STEPS", "6"))  # fewer steps = faster (10 is VoxCPM's default)
        rate = model.tts_model.sample_rate
        english = None
        if "en" not in VOX_LANGS:
            try:
                from kokoro import KPipeline

                english = KPipeline(lang_code="a")  # American English
            except Exception as err:  # not installed: VoxCPM2 speaks English too
                print("Kokoro (English voice) unavailable, VoxCPM2 speaks English:", repr(err), file=sys.stderr)
    except Exception as err:
        print(json.dumps({"error": repr(err)}), flush=True)
        return
    print(json.dumps({"rate": rate, "sample": bool(voices["km"][0]), "steps": steps, "english": "kokoro" if english else "voxcpm2"}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            text, lang = request["text"], request.get("lang") or ("km" if KHMER.search(request["text"]) else "en")
            if lang != "km" and english is not None:  # English: Kokoro, fast
                wave = np.concatenate([np.asarray(audio, dtype=np.float32) for _, _, audio in english(text, voice=ENGLISH_VOICE)])
                out_rate = 24000
            else:
                settings, prefix = voices["en" if lang == "en" else "km"]
                random.seed(7); np.random.seed(7); torch.manual_seed(7)  # the same voice every time
                args = {"cfg_value": 2.0, "inference_timesteps": steps, "seed": 7, **settings}
                wave = model.generate(text=prefix + text, **{k: v for k, v in args.items() if k in accepted})
                out_rate = rate
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            sf.write(path, wave, out_rate)
            print(json.dumps({"path": path}), flush=True)
        except Exception as err:
            print(json.dumps({"error": repr(err)}), flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--serve":
        serve(sys.argv[2])
    else:
        print(__doc__)
