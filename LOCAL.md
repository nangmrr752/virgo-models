# Train (and run) Virgo on your own GPU — Ubuntu

Works on a machine with an NVIDIA GPU, its driver (`nvidia-smi` works) and conda. An **RTX 4090 (24 GB)**
trains the **12B** in about 1–2 hours. The code is in **/opt/virgo-models**; trained models, logs and downloads go
in **/opt/virgo-data** (set `VIRGO_HOME` to change it).

## 1. Set up (once)
```bash
cd /opt && git clone https://github.com/nangmrr752/virgo-models    # not root? sudo chown "$USER" /opt/virgo-models after
bash /opt/virgo-models/scripts/local_setup.sh
```
It creates the conda environment `virgo`, installs PyTorch (CUDA 12.1) and the training packages, checks
the GPU, and asks for your Hugging Face token (write access) once. Accept the Gemma licenses on Hugging
Face first: [gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it), [gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it).

## 2. Train
```bash
tmux                                                          # keeps training going if you disconnect
bash /opt/virgo/virgo-models/scripts/train_local.sh bayon 27b  # → Virgo-1.0-Bayon (Gemma 3 27B, smarter than Angkor; ~3-5 h on a 4090)
bash /opt/virgo/virgo-models/scripts/train_local.sh angkor 12b    # → Virgo-1.0-Angkor-12B
bash /opt/virgo/virgo-models/scripts/train_local.sh hearing      # → Virgo-1.0-Angkor-Hearing: better Khmer hearing (all of Whisper, ~2-4 h; uploads only if its score improves)
bash /opt/virgo/virgo-models/scripts/train_local.sh voice        # → Virgo-1.0-Angkor-Voice: a steadier live voice (VoxCPM2 LoRA, ~1-2 h; not while another training runs)
```
Each run: pulls the latest code and data (plus the Gemma 27B examples if you made them), trains, scores
Virgo, and uploads the model privately to Hugging Face. Out of memory? add `--max-len 768`.
Leave tmux with **Ctrl+B, D**; come back with `tmux attach`. Watch the GPU: `watch -n 2 nvidia-smi`.

| Where | What |
|---|---|
| `/opt/virgo-data/out/<model>` | the trained model |
| `/opt/virgo-data/logs/` | full logs of every run (send the score part to Claude) |
| `/opt/virgo-data/hf-cache/` | downloaded base models (Gemma) |
| `/opt/virgo-models/env.sh` | `source /opt/virgo-models/env.sh` to work by hand in the right environment |

## 3. Run Virgo's server here (optional)
A 24 GB GPU runs the whole server (chat 12B, hearing, voice) at your own address:
```bash
source /opt/virgo-models/env.sh
export HF_TOKEN=... CF_TUNNEL_TOKEN=... VIRGO_API_KEY=...
bash scripts/start_server.sh
```
Stop any Kaggle/Colab/RunPod server first (they share the tunnel). Virgo is online while this machine is.

## Serving the 27B Bayon

Bayon 27B needs about 17 GB in 4 bits. The server adds it next to Angkor only when it fits
(`serve/fit.py`): a second GPU, or one GPU of 40 GB or more. Otherwise it serves Angkor alone.
