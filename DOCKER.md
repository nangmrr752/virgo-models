# Virgo with Docker Compose (your own GPU server)

Two containers: **virgo** (the Virgo server on your GPU: chat, hearing, voice, live voice) and
**tunnel** (Cloudflare Tunnel, so the website reaches it at your address, e.g. virgo.camksn.com).
Both restart by themselves, also after a reboot.

## Once: Docker with GPU access

```bash
curl -fsSL https://get.docker.com | sh
# NVIDIA Container Toolkit (lets containers use the GPU)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update && apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker && systemctl restart docker
docker run --rm --gpus all ubuntu nvidia-smi   # should show your GPU
```

## Start

```bash
cd /opt/virgo/virgo-models && git pull
cp .env.example .env && nano .env      # HF_TOKEN, CF_TUNNEL_TOKEN, VIRGO_API_KEY
docker compose up -d --build
docker compose logs -f virgo           # wait for "✅ Virgo server is live"
```

The first start downloads the models and sets up the voice (a while); later starts are quick, since
everything is kept in `/opt/virgo/docker`.

- Update: `git pull && docker compose up -d --build`
- Stop: `docker compose down`
- Stop any other Virgo server (RunPod, Kaggle, `start_server.sh`) that uses the same tunnel token.
- Don't train while it runs: the server uses the GPU.
- On one 24 GB GPU it serves Angkor, hearing and voice; Bayon 27B needs a second GPU or 40 GB+.
