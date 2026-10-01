# Virgo-1.0-Angkor — by KSN

Virgo 1.0 is KSN's own AI model family for [Virgo AI](https://virgoai.camksn.com). Each ability is a
small open model, tuned or set up to be Virgo: its name, its tone, English and Khmer.

| # | Ability | Virgo 1.0 part | Built on (open model) | Runs on |
|---|---|---|---|---|
| 1 | **Chat** | `chat/` | Gemma 3 4B + Virgo LoRA (trained here; 1B lighter, 12B smarter) | GPU, CPU (slow), Workers AI LoRA*, browser* |
| 2 | **Text to speech** | `speech/virgo_speech.py tts` | **Virgo-1.0-Angkor-Voice** on VoxCPM2 (Khmer, English and 28 more; designed with `speech/design_voice.ipynb`) · older fallbacks: Virgo's VITS Khmer voice / MMS-TTS, Kokoro-82M (English) | CPU, GPU |
| 3 | **Speech to text** | `speech/virgo_speech.py stt` | Whisper large-v3-turbo on a GPU (small on a CPU; small can be fine-tuned for Khmer here) | CPU, GPU |
| 4 | **Transcribe** | `speech/virgo_speech.py transcribe` | the same Whisper, long audio + timestamps + `.srt` | CPU, GPU |
| 5 | **Realtime speech to speech** | `realtime/` | VAD → Virgo STT → Virgo chat → Virgo TTS, streamed | GPU (best), CPU |
| 6 | **Images** | `image/` | SD-Turbo (1–4 steps) + optional Virgo style LoRA | GPU, CPU (slow) |
| 7 | **Videos** | `video/` | Wan 2.1 T2V 1.3B: real motion, 480p, ~5 s | NVIDIA GPU, 8 GB+ |

\* Where Virgo 1.0 is hosted is decided later. `serve/` runs every ability behind one API, so any
machine with Python (a GPU server, a PC, Hugging Face Spaces, a cloud VM) can host it.

Everything is described in [`virgo.json`](virgo.json).

## Train Virgo in one click

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nangmrr752/virgo-models/blob/main/chat/train_colab.ipynb)

Open the link → **Runtime → Change runtime type → T4 GPU** → **Runtime → Run all** → paste your Hugging Face token. Free, about 30–60 minutes.

**Smarter Virgo (12B):** open [`chat/train_kaggle.ipynb`](chat/train_kaggle.ipynb) on Kaggle and **Save & Run All**. It trains Virgo on Gemma 3 12B in 4 bits on a free T4 (about 2–4 hours) and saves `Virgo-1.0-Angkor-12B`. The server notebooks then use it automatically, and the 4B stays as a backup. Accept [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it)'s license first. It answers better, but writes about half as fast as 4B on a T4.

## Free TPU training (experimental)

[`chat/train_tpu_kaggle.ipynb`](chat/train_tpu_kaggle.ipynb) trains Virgo 12B on Kaggle's free **TPU v5e-8** (or on a Colab TPU, where smaller TPUs train the 4B)
(128 GB, full precision, its own weekly hours) with Keras + JAX, and saves a whole model,
**Virgo-1.0-Bayon** (Virgo-1.0-Bayon-4B on smaller TPUs), that the server loads with `HF_MODEL`. New: compare its answers with the
GPU-trained 12B before switching. The server itself stays on a GPU (the voice needs CUDA).

## Google Vertex AI (paid, faster, no free-tier limits)

On a Google Cloud **L4 GPU** (24 GB, 32 GB RAM) through **Colab Enterprise**: about 2–3× faster than a
free T4, no 12-hour or weekly limits, and it can run without a browser tab (Executions). New Google
Cloud accounts get $300 of free credit.

- **Train the 12B:** [`chat/train_vertex.ipynb`](chat/train_vertex.ipynb), about 1–2 hours (≈ $1–2).
- **Make Virgo smarter with more data:** [`chat/distill_vertex.ipynb`](chat/distill_vertex.ipynb): Gemma 3 27B
  writes and checks ~3,000 new Khmer/English examples (≈ 3–6 h); the training notebooks use them
  automatically. (Gemma's outputs are allowed for this; Gemini's are not.)
- **Run the server** (12B chat + hearing + Virgo's voice on one GPU, at virgo.camksn.com):
  [`serve/serve_vertex.ipynb`](serve/serve_vertex.ipynb), ≈ $0.85–1/hour while it runs.

Each notebook's first cell lists the one-time setup: GPU quota, a runtime template
(g2-standard-8 + L4, Spot to pay less), and secrets in Secret Manager. Stop the runtime when you
don't need it.

## Run Virgo for the website (chat + real-time voice)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nangmrr752/virgo-models/blob/main/serve/serve_colab.ipynb) — a free **test** server: it runs your trained Virgo-1.0-Angkor on Colab and prints `VIRGO_API_URL` and `VIRGO_API_KEY` for the Worker's secrets.

Colab's free GPU limit reached? Use **Kaggle** instead (free T4, about 30 GPU hours a week): open [`serve/serve_kaggle.ipynb`](serve/serve_kaggle.ipynb) on Kaggle (**Create → New notebook → File → Import notebook → GitHub**, or upload the file). It needs phone verification and an `HF_TOKEN` secret (**Add-ons → Secrets**); the steps are at the top of the notebook.

**A fixed address (e.g. `https://virgo.camksn.com`) instead of a new one each run:**
1. In Cloudflare, create a free tunnel: **Zero Trust → Networks → Tunnels → Create a tunnel → Cloudflared**. Give it the public hostname `virgo.camksn.com` → HTTP `localhost:8000`.
2. Add two notebook secrets:
   - `CF_TUNNEL_TOKEN`: the tunnel's token.
   - `VIRGO_API_KEY`: a long password you choose.
3. Set the Worker secrets once: `VIRGO_API_URL=https://virgo.camksn.com` and the same `VIRGO_API_KEY`.

After that, running the notebook is all it takes. Without those secrets, the notebooks use a random `trycloudflare.com` address, as before.

For a permanent server on a rented GPU, see **[RUNPOD.md](RUNPOD.md)**: one start command runs everything
(models, voice, your Cloudflare address) and restarts the server if it stops. By hand:

```bash
git clone https://github.com/nangmrr752/virgo-models && cd virgo-models
pip install -r requirements-serve.txt
huggingface-cli download <your-name>/Virgo-1.0-Angkor --local-dir chat/out/virgo-1.0-chat-lora
export VIRGO_API_KEY=<a long random key> VIRGO_PRELOAD=chat,speech
uvicorn serve.app:app --host 0.0.0.0 --port 8000
```

The website uses `POST /v1/chat` for chat and the `/v1/realtime` WebSocket for **Virgo Live**: it listens (voice activity detection + Whisper), answers with Virgo-1.0-Angkor, and speaks each sentence as soon as it's written (Khmer: Virgo's own voice once trained, else MMS; English: Kokoro, or MMS English where Kokoro can't be installed). Talking over Virgo interrupts it.

## What "our own model" means here

Training a model from zero needs huge data and thousands of GPUs. Virgo 1.0 does what most
companies do: it starts from strong open models and **teaches them to be Virgo**.
- **Chat:** trained with our own examples (`chat/data/`), so it knows it's Virgo by KSN and answers in Virgo's style, in English and Khmer.
- **Speech to text:** fine-tuned on Khmer speech, so it understands Khmer better.
- **Voices, images and video:** ready-made open models wrapped as Virgo, with room to tune later (voice cloning, a Virgo image style).

Your code and training data live in this repo. Trained weights are large, so they go to Hugging Face
(private repos are fine) or Cloudflare R2 (bucket `virgo-models`), not into git.

## Quick start

```bash
# 1. PyTorch for your machine first: https://pytorch.org/get-started/
pip install -r requirements.txt

# 2. Check the training data, train Virgo chat (a free Colab T4 GPU: chat/train_colab.ipynb), and score it
python scripts/check_data.py
python chat/train.py                 # Gemma 3 4B in 4 bits; --base google/gemma-3-1b-it for a CPU
python chat/evaluate.py              # scores Virgo on chat/eval/questions.jsonl

# 3. Try each ability
python chat/chat.py
python speech/virgo_speech.py tts "សួស្តី ខ្ញុំឈ្មោះ Virgo" hello.wav
python speech/virgo_speech.py stt hello.wav --language khmer
python speech/virgo_speech.py transcribe meeting.mp3 --srt meeting.srt
python image/virgo_image.py "Angkor Wat at sunrise, watercolor" angkor.png
python video/virgo_video.py "a horse running on the beach at sunset" horse.mp4   # GPU
python realtime/virgo_realtime.py                                                # ws://localhost:8765

# 4. Or run everything as one API
uvicorn serve.app:app --host 0.0.0.0 --port 8000        # or: docker build -t virgo . && docker run -p 8000:8000 virgo
```

The API (`serve/app.py`):

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/v1/chat` | `{"messages": [...]}` | `{"reply": "..."}` |
| POST | `/v1/tts` | `{"text": "...", "voice": "af_heart"}` | `audio/wav` |
| POST | `/v1/stt` | audio file (`file`), `?language=khmer` | `{"text": "..."}` |
| POST | `/v1/transcribe` | audio file (`file`), `?srt=1` | segments, or `.srt` |
| POST | `/v1/images` | `{"prompt": "...", "steps": 2}` | `image/png` |
| POST | `/v1/videos` | `{"prompt": "...", "seconds": 5}` | `video/mp4` |
| WS | `/v1/realtime` | 16 kHz 16-bit PCM audio | JSON events + WAV audio per sentence |
| GET | `/v1/models` | | `virgo.json` |

Set `VIRGO_API_KEY` to require `Authorization: Bearer <key>` (WebSocket: `?key=<key>`).

## Making Virgo better

- **Chat:** add more examples to `chat/data/` (one JSON line per conversation, same format as
  `virgo_chat.jsonl`). A few hundred to a few thousand good examples make a big difference: Virgo's
  identity, common questions from your users, and plenty of natural Khmer. Run `scripts/check_data.py`, then train again.
- **Virgo's own Khmer hearing:** open [`speech/train_hearing_kaggle.ipynb`](speech/train_hearing_kaggle.ipynb) on Kaggle and
  **Save & Run All**. It fine-tunes Whisper large-v3-turbo with LoRA on OpenSLR 42 (plus FLEURS when it
  loads) in about 3–5 hours, prints the Khmer error rate before and after, and saves `Virgo-1.0-Angkor-Hearing`
  if it improved. The server notebooks then use it automatically. Add your own recordings with `--extra`
  (audio files + `metadata.csv` with `file_name,sentence`).
- **A Virgo image style:** train a LoRA on 20–50 images in your style with diffusers'
  `train_text_to_image_lora.py` on SD-Turbo's base, then pass `--lora` to `image/virgo_image.py`.
- **Virgo's voice (Virgo-1.0-Angkor-Voice):** open [`speech/design_voice.ipynb`](speech/design_voice.ipynb) on Kaggle or
  Colab, describe a voice in words, listen to a few takes and pick one; it's saved to Hugging Face and the
  server notebooks use it for every language. An optional LoRA step makes it steadier (use a real speaker's
  recordings only with their permission).
- **Older Khmer voice (VITS):** open [`speech/train_voice_kaggle.ipynb`](speech/train_voice_kaggle.ipynb) on Kaggle and
  **Save & Run All**. It trains a VITS voice from scratch on OpenSLR 42 (commercial-safe), about 11 hours per run.
  Each run continues the last one, and a clear voice takes 3–5 runs. The server notebooks then use it
  for Khmer automatically. Add 1–2 hours of one speaker's recordings (`--extra`) for a unique voice.

## Licenses (check before commercial use)

Virgo's own code here is yours. Each base model keeps its own license:

| Model | License |
|---|---|
| Gemma 3 (1B, 4B, 12B) | [Gemma Terms of Use](https://ai.google.dev/gemma/terms): accept on Hugging Face before downloading |
| Whisper | MIT |
| Kokoro-82M | Apache 2.0 |
| VoxCPM2 (Virgo-1.0-Angkor-Voice's base) | Apache 2.0: free for commercial use; credit "built on VoxCPM2 by OpenBMB" |
| MMS-TTS (Khmer) | CC BY-NC 4.0: **non-commercial**. Only the fallback until Virgo's own voice is trained. |
| Virgo's Khmer voice (trained here) | Your model. Training data OpenSLR 42 is CC BY-SA 4.0: credit "Khmer speech data by Google (OpenSLR 42)". Code: coqui-tts (MPL 2.0). |
| SD-Turbo | Stability AI license: check its model card for your use |
| Wan 2.1 | Apache 2.0 |

Licenses change: read each model card before you ship.
