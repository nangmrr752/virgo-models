"""Virgo's hearing test: how well speech-to-text models hear Khmer, measured as CER (character error rate,
lower is better; Khmer has no spaces between words, so characters are compared, not words).

    python speech/eval_stt.py --data fleurs                     # FLEURS Khmer test set (public, not used in training)
    python speech/eval_stt.py --data /opt/virgo/stt-test        # your own clips: audio files + metadata.csv
    python speech/eval_stt.py --data fleurs --models \\
        whisper:virgoai/Angkor-1.0-STT \\
        whisper:openai/whisper-large-v3-turbo \\
        qwen3asr:seanghay/Qwen3-ASR-0.6B-Khmer \\
        omni:omniASR_LLM_7B \\
        server:http://127.0.0.1:8088

Models (one after another, each freed from the GPU before the next):
  whisper:<repo>   any Whisper model (Angkor-1.0-STT, openai/whisper-*, Khmer fine-tunes) with transformers
  qwen3asr:<repo>  Qwen3-ASR models (pip install qwen-asr), e.g. seanghay/Qwen3-ASR-0.6B-Khmer
  omni:<card>      Meta Omnilingual ASR (pip install omnilingual-asr), e.g. omniASR_LLM_7B, omniASR_LLM_1B
  hf:<repo>        any other transformers speech-recognition model (trust_remote_code)
  server:<url>     the running Virgo server's /v1/stt (Virgo's real two-ear hearing; key: VIRGO_API_KEY)

Your own clips: a folder of .wav/.mp3/.m4a/.ogg/.flac files with metadata.csv (file_name,sentence), or a .txt
file next to each clip holding what it says. Mix in clips with English words to test mixed speech.
Report: <data>/logs/stt-<date>.json (every clip's text and errors), and a table at the end.
"""
import argparse
import csv
import gc
import io
import json
import os
import re
import sys
import time
import unicodedata

RATE = 16000
AUDIO = (".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm")
KHMER_DIGITS = str.maketrans("០១២៣៤៥៦៧៨៩", "0123456789")
# Punctuation and invisible marks that don't change what was said (Khmer ។ ៕ ៖, zero-width spaces…).
PUNCT = re.compile(r"[\s​‌‍﻿.,!?;:\"'()\[\]{}«»“”‘’\-–—…។៕៖/\\|*_#]+")


def normalize(text):
    text = unicodedata.normalize("NFC", str(text or "")).lower().translate(KHMER_DIGITS)
    return PUNCT.sub("", text)


def edits(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# ---------- data ----------

def own_clips(folder):
    index = os.path.join(folder, "metadata.csv")
    rows = []
    if os.path.exists(index):
        with open(index, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = row.get("file_name") or row.get("file") or row.get("path")
                text = row.get("sentence") or row.get("text") or row.get("transcription")
                if name and text and os.path.exists(os.path.join(folder, name)):
                    rows.append((os.path.join(folder, name), text.strip()))
    else:
        for name in sorted(os.listdir(folder)):
            path = os.path.join(folder, name)
            txt = os.path.splitext(path)[0] + ".txt"
            if name.lower().endswith(AUDIO) and os.path.exists(txt):
                rows.append((path, open(txt, encoding="utf-8").read().strip()))
    if not rows:
        raise SystemExit(f"No clips in {folder}: add audio files with metadata.csv (file_name,sentence) or a .txt per clip")
    return rows


def fleurs_test(work, limit):
    """FLEURS Khmer *test* split (Angkor-1.0-STT trained on the train split), saved once as wav files."""
    import soundfile as sf

    folder = os.path.join(work, "fleurs-test")
    index = os.path.join(folder, "metadata.csv")
    if not os.path.exists(index):
        from datasets import Audio, load_dataset

        try:
            rows = load_dataset("google/fleurs", "km_kh", split="test")
        except Exception:
            rows = load_dataset("google/fleurs", "km_kh", split="test", trust_remote_code=True)
        rows = rows.cast_column("audio", Audio(decode=False))
        os.makedirs(folder, exist_ok=True)
        with open(index + ".part", "w", encoding="utf-8", newline="") as f:
            out = csv.writer(f)
            out.writerow(["file_name", "sentence"])
            for i, row in enumerate(rows):
                audio = row["audio"]
                source = io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"]
                data, rate = sf.read(source)
                name = f"{i:05d}.wav"
                sf.write(os.path.join(folder, name), data, rate)
                out.writerow([name, row.get("transcription") or row.get("raw_transcription") or ""])
        os.replace(index + ".part", index)
        print(f"FLEURS Khmer test: saved to {folder}")
    return own_clips(folder)[:limit]


# ---------- models ----------

def free():
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def whisper_model(repo, language):
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("automatic-speech-recognition", model=repo, device=device,
                    torch_dtype=torch.float16 if device == 0 else torch.float32)
    kwargs = {"generate_kwargs": {"language": language, "task": "transcribe"}} if language else {}
    return lambda path: pipe(path, chunk_length_s=30, **kwargs)["text"]


def hf_model(repo, language):
    import torch
    from transformers import pipeline

    pipe = pipeline("automatic-speech-recognition", model=repo, trust_remote_code=True,
                    device=0 if torch.cuda.is_available() else -1)
    return lambda path: pipe(path)["text"]


def qwen3asr_model(repo, language):
    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError:
        raise SystemExit("Qwen3-ASR needs its package: python -m pip install -U qwen-asr")
    import torch

    model = Qwen3ASRModel.from_pretrained(repo, dtype=torch.bfloat16, device_map="cuda:0" if torch.cuda.is_available() else "cpu")

    def run(path):
        out = model.transcribe(audio=path, language=None)
        out = out[0] if isinstance(out, (list, tuple)) else out
        return getattr(out, "text", out if isinstance(out, str) else str(out))
    return run


def omni_model(card, language):
    try:
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    except ImportError:
        raise SystemExit("Meta Omnilingual ASR needs its package: python -m pip install -U omnilingual-asr")
    pipe = ASRInferencePipeline(model_card=card)
    return lambda path: pipe.transcribe([path], lang=["khm_Khmr"], batch_size=1)[0]


def server_model(url, language):
    import urllib.request
    import uuid

    key = os.environ.get("VIRGO_API_KEY", "")

    def run(path):
        boundary = uuid.uuid4().hex
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{os.path.basename(path)}\"\r\n"
                f"Content-Type: application/octet-stream\r\n\r\n").encode() + open(path, "rb").read() + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(url.rstrip("/") + "/v1/stt", data=body, method="POST",
                                     headers={"content-type": f"multipart/form-data; boundary={boundary}",
                                              **({"authorization": f"Bearer {key}"} if key else {})})
        with urllib.request.urlopen(req, timeout=300) as res:
            return json.loads(res.read()).get("text", "")
    return run


ENGINES = {"whisper": whisper_model, "hf": hf_model, "qwen3asr": qwen3asr_model, "omni": omni_model, "server": server_model}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="fleurs", help='"fleurs" (FLEURS Khmer test) or a folder of your own clips')
    p.add_argument("--models", nargs="+", default=["whisper:virgoai/Angkor-1.0-STT", "whisper:openai/whisper-large-v3-turbo"])
    p.add_argument("--limit", type=int, default=200, help="at most this many clips")
    p.add_argument("--language", default="khmer", help="the language told to Whisper models (empty: they guess)")
    p.add_argument("--report", default="")
    args = p.parse_args()

    home = os.environ.get("VIRGO_HOME", os.path.join(os.path.dirname(__file__), "..", "out"))
    clips = fleurs_test(os.path.join(home, "stt-test"), args.limit) if args.data == "fleurs" else own_clips(args.data)[: args.limit]
    print(f"{len(clips)} clips from {args.data}", flush=True)

    results = {}
    for spec in args.models:
        kind, _, name = spec.partition(":")
        if kind not in ENGINES or not name:
            raise SystemExit(f"Unknown model {spec!r}: use whisper:/qwen3asr:/omni:/hf:/server: (see --help)")
        print(f"\n== {spec} ==", flush=True)
        try:
            run = ENGINES[kind](name, args.language)
        except SystemExit as err:
            print("  skipped:", err)
            continue
        except Exception as err:
            print("  couldn't load:", type(err).__name__, str(err)[:300])
            continue
        rows, errs, chars, began = [], 0, 0, time.time()
        for i, (path, ref) in enumerate(clips, 1):
            try:
                hyp = run(path) or ""
            except Exception as err:
                print(f"  {os.path.basename(path)}: failed ({type(err).__name__})")
                hyp = ""
            a, b = normalize(ref), normalize(hyp)
            e = edits(a, b)
            errs, chars = errs + e, chars + max(1, len(a))
            rows.append({"file": os.path.basename(path), "ref": ref, "hyp": hyp, "cer": round(e / max(1, len(a)), 4)})
            if i % 20 == 0 or i == len(clips):
                print(f"  {i}/{len(clips)}  CER so far {errs / chars:.1%}", flush=True)
        seconds = (time.time() - began) / max(1, len(clips))
        results[spec] = {"cer": errs / chars, "clips": len(clips), "seconds_per_clip": round(seconds, 2), "rows": rows}
        del run
        free()

    if not results:
        raise SystemExit("No model could run.")
    print("\n== Khmer hearing: CER (lower is better) ==")
    for spec, r in sorted(results.items(), key=lambda kv: kv[1]["cer"]):
        print(f"  {r['cer']:6.1%}   {r['seconds_per_clip']:5.2f} s/clip   {spec}")
    report = args.report or os.path.join(home, "logs", f"stt-{time.strftime('%Y%m%d-%H%M')}.json")
    os.makedirs(os.path.dirname(report), exist_ok=True)
    json.dump({"data": args.data, "results": results}, open(report, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("Report:", report)


if __name__ == "__main__":
    sys.exit(main())
