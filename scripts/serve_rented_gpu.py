"""The Virgo server on a rented GPU (started by scripts/start_server.sh): the same steps as the server
notebooks, without a notebook. Downloads Virgo's models, sets up Virgo's voice, connects your
Cloudflare Tunnel, then runs the server and restarts it if it ever stops.
"""
import os
import subprocess
import sys
import time
import urllib.request

import torch

WORKSPACE = os.environ.get("WORKSPACE", "/workspace")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)


def need(name):
    value = os.environ.get(name, "").strip()
    if not value:
        sys.exit(f"❌ Set the {name} environment variable (see RUNPOD.md).")
    return value


def link_to_workspace(folder):
    """Keeps a model folder on the persistent volume (WORKSPACE), linked into the code folder."""
    target = os.path.join(WORKSPACE, "virgo-data", folder)
    os.makedirs(target, exist_ok=True)
    if not os.path.islink(folder):
        os.makedirs(os.path.dirname(folder), exist_ok=True)
        if os.path.isdir(folder):
            subprocess.run(["rm", "-rf", folder])
        os.symlink(target, folder)
    return folder


def has_files(folder):
    return os.path.isdir(folder) and any(os.scandir(folder))


# ---------- 1. Models ----------
assert torch.cuda.is_available(), "❌ No NVIDIA GPU found: pick a GPU pod."
gpus = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
print("✅ GPUs:", ", ".join(f"{g.name} ({g.total_memory / 1e9:.0f} GB)" for g in gpus))
big_gpu = gpus[0].total_memory > 20e9

from huggingface_hub import HfApi, login, snapshot_download  # noqa: E402

login(token=need("HF_TOKEN"))
api = HfApi()
me = api.whoami()["name"]

# snapshot_download only fetches what changed, so a restart picks up newly trained models.
# VIRGO_MAIN=bayon: Bayon instead of Angkor (Bayon 27B and Angkor don't both fit one 24 GB GPU).
bayon_only = os.environ.get("VIRGO_MAIN", "angkor").lower() == "bayon"
ADAPTER = link_to_workspace("chat/out/virgo-1.0-chat-lora")
if not bayon_only:
    repo = os.environ.get("HF_MODEL") or next(
        (r for r in (f"{me}/{n}" for n in ("Virgo-Angkor-1.0-12B", "Virgo-1.0-Angkor-12B", "Virgo-Angkor-1.0-4B", "Virgo-1.0-Angkor"))
         if api.file_exists(r, "adapter_config.json")),
        f"{me}/Virgo-Angkor-1.0-4B")  # standard names first (scripts/names.py), then the old ones
    print("⬇️ Downloading", repo)
    snapshot_download(repo, local_dir=ADAPTER)
os.environ["VIRGO_CHAT_MODELS"] = "" if bayon_only else f"virgo-1.0-angkor={ADAPTER}"

if bayon_only or os.environ.get("SERVE_BAYON", "1") != "0":
    has_bayon = False
    for bayon_repo in (f"{me}/{n}" for n in ("Virgo-Bayon-1.0-27B", "Virgo-1.0-Bayon", "Virgo-Bayon-1.0-12B", "Virgo-1.0-Bayon-12B",
                                              "Virgo-Bayon-1.0-4B", "Virgo-1.0-Bayon-4B")):
        try:
            has_bayon = api.file_exists(bayon_repo, "config.json") or api.file_exists(bayon_repo, "adapter_config.json")
        except Exception:
            has_bayon = False
        if has_bayon:
            break
    sys.path.append(os.path.join(ROOT, "serve"))
    from fit import bayon_fits

    if bayon_only and not has_bayon:
        sys.exit("❌ VIRGO_MAIN=bayon, but there's no Virgo-Bayon-1.0 on your Hugging Face account yet.")
    if has_bayon and (bayon_only or bayon_fits(bayon_repo)):
        BAYON = link_to_workspace("chat/out/virgo-1.0-bayon")
        print("⬇️ Checking", bayon_repo)
        snapshot_download(bayon_repo, local_dir=BAYON)
        os.environ["VIRGO_CHAT_MODELS"] = ",".join(filter(None, [os.environ["VIRGO_CHAT_MODELS"], f"virgo-1.0-bayon={BAYON}"]))
        print("✅ Also serving", bayon_repo.split("/")[1], "at the same address")
    elif has_bayon:
        print(f"ℹ️ {bayon_repo} won't fit next to Angkor on these GPUs: serving Angkor only.")

# The chat models' base (e.g. Gemma 3 27B for Bayon, ~55 GB the first time), downloaded here with a
# progress bar, so the server's loading step isn't a long silent wait.
import json  # noqa: E402

for spec in filter(None, os.environ["VIRGO_CHAT_MODELS"].split(",")):
    cfg = os.path.join(spec.split("=", 1)[1], "adapter_config.json")
    if os.path.exists(cfg):
        base = json.load(open(cfg)).get("base_model_name_or_path")
        if base and not os.path.isdir(base):
            print("⬇️ Getting the base model", base, "(only the first time)")
            snapshot_download(base, allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"])

HEARING = link_to_workspace("speech/out/virgo-1.0-stt")
if True:
    try:
        hearing_repo = next((f"{me}/{n}" for n in ("Virgo-Angkor-1.0-STT", "Virgo-1.0-Angkor-Hearing")
                             if api.file_exists(f"{me}/{n}", "config.json")), None)
        if hearing_repo:
            print("⬇️ Downloading Virgo's Khmer hearing:", hearing_repo)
            snapshot_download(hearing_repo, local_dir=HEARING)
    except Exception as err:
        print("Virgo's own hearing not used:", err)

# ---------- 2. Virgo's voice (VoxCPM2 for Khmer, Kokoro for English), in its own environment ----------
if os.environ.get("USE_VIRGO_VOICE", "1") != "0":
    env_dir = os.path.join(WORKSPACE, "vox-env")
    python = os.path.join(env_dir, "bin", "python")
    if not os.path.exists(python):
        print("⏳ Installing Virgo's voice (first start only, a few minutes)...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
        subprocess.run(["uv", "venv", "-q", "--python", "3.11", env_dir], check=True)
        r = subprocess.run(["uv", "pip", "install", "-q", "--python", python, "git+https://github.com/OpenBMB/VoxCPM", "soundfile", "huggingface_hub"])
        if r.returncode:
            subprocess.run(["uv", "pip", "install", "-q", "--python", python, "voxcpm", "soundfile", "huggingface_hub"], check=True)
    if subprocess.run([python, "-c", "import kokoro"], capture_output=True).returncode:
        subprocess.run(["uv", "pip", "install", "-q", "--python", python, "kokoro>=0.9"])
    # Kokoro's English needs spaCy's small English model; install it now instead of on first use.
    if subprocess.run([python, "-c", "import en_core_web_sm"], capture_output=True).returncode:
        subprocess.run(["uv", "pip", "install", "-q", "--python", python,
                        "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"])
    # The newest PyTorch needs a newer NVIDIA driver than many machines have (e.g. 535): then the
    # voice runs on the CPU, far too slowly. Use a CUDA 12.4 build when the GPU isn't seen.
    if subprocess.run([python, "-c", "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"], capture_output=True).returncode:
        print("⏳ Installing a PyTorch build this GPU driver supports, for the voice...")
        subprocess.run(["uv", "pip", "install", "-q", "--python", python, "torch==2.6.0", "torchaudio==2.6.0",
                        "--index-url", "https://download.pytorch.org/whl/cu124"])
        subprocess.run(["uv", "pip", "install", "-q", "--python", python, "torchcodec==0.2.*"])
    subprocess.run([python, "-c", "from huggingface_hub import snapshot_download; snapshot_download('openbmb/VoxCPM2')"], check=True)
    os.environ["VIRGO_VOX_PYTHON"] = python
    # Which voice: Virgo-Bakong-{VIRGO_LIVE_VERSION}-TTS when it exists (train_local.sh voice bakong),
    # else Bayon's own voice when Bayon is the chat model (voice bayon), else Virgo-Angkor-1.0-TTS.
    # Standard names first (scripts/names.py), then the old ones.
    # VIRGO_VOICE_REPO=<name> picks one by hand.
    def has_voice(name):
        try:
            return api.file_exists(f"{me}/{name}", "voice.wav")
        except Exception:
            return False

    live_version = os.environ.get("VIRGO_LIVE_VERSION", "2.0").strip()
    candidates = [os.environ.get("VIRGO_VOICE_REPO", ""), f"Virgo-Bakong-{live_version}-TTS", f"Virgo-Bakong-{live_version}-Voice",
                  *(["Virgo-Bayon-1.0-TTS", "Virgo-1.0-Bayon-Voice"] if bayon_only else []), "Virgo-Angkor-1.0-TTS", "Virgo-1.0-Angkor-Voice"]
    voice_repo = f"{me}/" + next((c for c in candidates if c and has_voice(c)), "Virgo-Angkor-1.0-TTS")
    os.environ["VIRGO_VOX_REPO"] = voice_repo
    # Each voice keeps its own folder, so switching voices never mixes their files.
    os.environ["VIRGO_VOX_DIR"] = link_to_workspace("speech/out/" + voice_repo.split("/")[1].lower())
    if "Bayon" in voice_repo:  # Bayon's default voice is a man's: English (Kokoro) matches
        os.environ.setdefault("VIRGO_ENGLISH_VOICE", "am_michael")
    print("🗣️ Voice sample:", voice_repo)
    if len(gpus) > 1:
        os.environ["VIRGO_VOX_DEVICE"] = "cuda:1"
    if not big_gpu and len(gpus) == 1:  # a 16 GB GPU: squeeze chat into 4 bits so the voice fits
        os.environ["VIRGO_CHAT_4BIT"] = "1"
        os.environ["VIRGO_VOX_OPTIMIZE"] = "0"
    print("✅ Virgo's voice is set up")

# ---------- 3. Your address (Cloudflare Tunnel) ----------
# VIRGO_TUNNEL=external: the tunnel runs elsewhere (its own container in docker-compose.yml).
own_tunnel = os.environ.get("VIRGO_TUNNEL") != "external"
tunnel_token = need("CF_TUNNEL_TOKEN") if own_tunnel else None
key = need("VIRGO_API_KEY")
cloudflared = os.path.join(WORKSPACE, "cloudflared")
if own_tunnel and not os.path.exists(cloudflared):
    urllib.request.urlretrieve("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", cloudflared)
    os.chmod(cloudflared, 0o755)
log_dir = os.path.join(WORKSPACE, "logs")
os.makedirs(log_dir, exist_ok=True)


def start_tunnel():
    if not own_tunnel:
        return None
    # The token goes in the environment, never on the command line.
    return subprocess.Popen([cloudflared, "tunnel", "--no-autoupdate", "run"], env={**os.environ, "TUNNEL_TOKEN": tunnel_token},
                            stdout=open(os.path.join(log_dir, "tunnel.log"), "a"), stderr=subprocess.STDOUT)


def start_server():
    env = {**os.environ, "VIRGO_API_KEY": key, "VIRGO_PRELOAD": "chat,speech"}
    return subprocess.Popen([sys.executable, "-m", "uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"], env=env)


# ---------- 4. Run, and restart whatever stops ----------
tunnel, server = start_tunnel(), start_server()
print("⏳ Loading Virgo (a few minutes)...")
for _ in range(600):
    time.sleep(2)
    try:
        urllib.request.urlopen("http://localhost:8000/v1/models", timeout=5)
        print("✅ Virgo server is live at your Cloudflare address (e.g. https://virgo.camksn.com)")
        print("   Models:", os.environ["VIRGO_CHAT_MODELS"].replace(ROOT + "/", ""))
        break
    except Exception:
        if server.poll() is not None:
            break
while True:
    time.sleep(15)
    if tunnel is not None and tunnel.poll() is not None:
        print("⚠️ The tunnel stopped: restarting it")
        tunnel = start_tunnel()
    if server.poll() is not None:
        print(f"⚠️ The Virgo server stopped (exit {server.returncode}): restarting it in 10 s")
        time.sleep(10)
        server = start_server()
