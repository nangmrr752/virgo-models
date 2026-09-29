# Virgo 1.0 — by KSN

Virgo 1.0 is KSN's own AI model family for [Virgo AI](https://virgoai.camksn.com). Each ability is a
small open model, tuned or set up to be Virgo: its name, its tone, English and Khmer.

| # | Ability | Virgo 1.0 part | Built on (open model) | Runs on |
|---|---|---|---|---|
| 1 | **Chat** | `chat/` | Gemma 3 4B + Virgo LoRA (trained here; 1B lighter, 12B smarter) | GPU, CPU (slow), Workers AI LoRA*, browser* |
| 2 | **Text to speech** | `speech/virgo_speech.py tts` | MMS-TTS Khmer · Kokoro-82M (English) | CPU, GPU |
| 3 | **Speech to text** | `speech/virgo_speech.py stt` | Whisper small, fine-tuned for Khmer (trained here) | CPU, GPU |
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
- **Khmer speech to text:** `python speech/finetune_stt.py` (FLEURS Khmer by default). Add your own
  recordings with `--extra folder/` (audio files + `metadata.csv` with `file_name,sentence`).
- **A Virgo image style:** train a LoRA on 20–50 images in your style with diffusers'
  `train_text_to_image_lora.py` on SD-Turbo's base, then pass `--lora` to `image/virgo_image.py`.
- **A Virgo voice:** fine-tuning MMS-TTS on 1–2 hours of one speaker's Khmer recordings gives Virgo its
  own Khmer voice (see the `finetune-hf-vits` project).

## Licenses (check before commercial use)

Virgo's own code here is yours. Each base model keeps its own license:

| Model | License |
|---|---|
| Gemma 3 (1B, 4B, 12B) | [Gemma Terms of Use](https://ai.google.dev/gemma/terms): accept on Hugging Face before downloading |
| Whisper | MIT |
| Kokoro-82M | Apache 2.0 |
| MMS-TTS (Khmer) | CC BY-NC 4.0: **non-commercial**. For a paid product, train or pick a commercially licensed Khmer voice. |
| SD-Turbo | Stability AI license: check its model card for your use |
| Wan 2.1 | Apache 2.0 |

Licenses change: read each model card before you ship.
