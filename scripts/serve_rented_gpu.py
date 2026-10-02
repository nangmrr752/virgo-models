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
        (r for r in (f"{me}/Virgo-1.0-Angkor-12B", f"{me}/Virgo-1.0-Angkor") if api.file_exists(r, "adapter_config.json")),
        f"{me}/Virgo-1.0-Angkor")
    print("⬇️ Downloading", repo)
    snapshot_download(repo, local_dir=ADAPTER)
os.environ["VIRGO_CHAT_MODELS"] = "" if bayon_only else f"virgo-1.0-angkor={ADAPTER}"

if bayon_only or os.environ.get("SERVE_BAYON", "1") != "0":
    has_bayon = False
    for bayon_repo in (f"{me}/Virgo-1.0-Bayon", f"{me}/Virgo-1.0-Bayon-12B", f"{me}/Virgo-1.0-Bayon-4B"):
        try:
            has_bayon = api.file_exists(bayon_repo, "config.json") or api.file_exists(bayon_repo, "adapter_config.json")
        except Exception:
            has_bayon = False
        if has_bayon:
            break
    sys.path.append(os.path.join(ROOT, "serve"))
    from fit import bayon_fits

    if bayon_only and not has_bayon:
        sys.exit("❌ VIRGO_MAIN=bayon, but there's no Virgo-1.0-Bayon on your Hugging Face account yet.")
    if has_bayon and (bayon_only or bayon_fits(bayon_repo)):
        BAYON = link_to_workspace("chat/out/virgo-1.0-bayon")
        print("⬇️ Checking Virgo-1.0-Bayon")
        snapshot_download(bayon_repo, local_dir=BAYON)
        os.environ["VIRGO_CHAT_MODELS"] = ",".join(filter(None, [os.environ["VIRGO_CHAT_MODELS"], f"virgo-1.0-bayon={BAYON}"]))
        print("✅ Also serving Virgo-1.0-Bayon at the same address")
    elif has_bayon:
        print(f"ℹ️ {bayon_repo} won't fit next to Angkor on these GPUs: serving Angkor only.")

HEARING = link_to_workspace("speech/out/virgo-1.0-stt")
if True:
    try:
        if api.file_exists(f"{me}/Virgo-1.0-Angkor-Hearing", "config.json"):
            print("⬇️ Downloading Virgo's Khmer hearing")
            snapshot_download(f"{me}/Virgo-1.0-Angkor-Hearing", local_dir=HEARING)
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
    # The newest PyTorch needs a newer NVIDIA driver than many machines have (e.g. 535): then the
    # voice runs on the CPU, far too slowly. Use a CUDA 12.4 build when the GPU isn't seen.
    if subprocess.run([python, "-c", "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"], capture_output=True).returncode:
        print("⏳ Installing a PyTorch build this GPU driver supports, for the voice...")
        subprocess.run(["uv", "pip", "install", "-q", "--python", python, "torch==2.6.0", "torchaudio==2.6.0",
                        "--index-url", "https://download.pytorch.org/whl/cu124"])
        subprocess.run(["uv", "pip", "install", "-q", "--python", python, "torchcodec==0.2.*"])
    subprocess.run([python, "-c", "from huggingface_hub import snapshot_download; snapshot_download('openbmb/VoxCPM2')"], check=True)
    os.environ["VIRGO_VOX_PYTHON"] = python
    os.environ["VIRGO_VOX_REPO"] = f"{me}/Virgo-1.0-Angkor-Voice"
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
