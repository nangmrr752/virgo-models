# Run Virgo on RunPod (a rented GPU, no notebook)

One pod runs Virgo's whole server (chat, Khmer hearing, Virgo's voice, Virgo Live) at **your own
address** (virgo.camksn.com, through your Cloudflare Tunnel). Nothing changes on the website.

## 1. Create the pod (once)
1. [runpod.io](https://www.runpod.io) → add credit ($25–50 is plenty to start).
2. **Pods → Deploy**. GPU with **24 GB** so everything fits: **RTX 4090**, **L4**, **RTX A5000** or **RTX 3090**
   (Community Cloud is cheaper). For Angkor + Bayon together, 48 GB (e.g. A6000 / L40S) or 2 GPUs.
3. Template: **RunPod PyTorch** (2.4 or newer).
4. **Volume disk: 100 GB**, mounted at `/workspace` (models stay there between restarts).
   Container disk: 30 GB.
5. **Environment variables** (use RunPod *Secrets* for the tokens if you like):

   | Name | Value |
   |---|---|
   | `HF_TOKEN` | your Hugging Face token |
   | `CF_TUNNEL_TOKEN` | your Cloudflare Tunnel token (the same one as in the notebooks) |
   | `VIRGO_API_KEY` | the same password the website uses |

6. **Container start command:**
   ```
   bash -c "git clone https://github.com/nangmrr752/virgo-models /workspace/virgo-models 2>/dev/null; cd /workspace/virgo-models && git pull && bash scripts/start_server.sh"
   ```
7. Deploy. The first start takes ~10–15 minutes (downloads); later starts a few minutes.

## 2. Check it
Pod → **Logs**: look for `✅ Virgo server is live` and `🗣️ Voice: Virgo-1.0-Angkor-Voice`. Then chat on
virgoai.camksn.com. Server and tunnel logs are also in `/workspace/logs`.

## 3. Save money
- **Stop** the pod when nobody needs Virgo: you only pay for the volume (~$0.10/GB-month), and the website
  answers with GPT-6 Luna meanwhile. **Start** it again and Virgo is back in a few minutes.
- Don't run a Kaggle/Colab server at the same time (they'd share the tunnel).

## Options (environment variables)
- `HF_MODEL=<you>/Virgo-1.0-Angkor` — pick a chat model (default: your 12B if trained, else the 4B).
- `SERVE_BAYON=0` — don't serve Virgo-1.0-Bayon even if it exists.
- `USE_VIRGO_VOICE=0` — skip Virgo's VoxCPM2 voice (older voices).

New training (a new Virgo-1.0-Angkor, Bayon or voice on Hugging Face) is picked up on the next restart.
