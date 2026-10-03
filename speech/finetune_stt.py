"""Fine-tune Virgo's hearing (speech to text, Whisper) for Khmer.

    python speech/finetune_stt.py --push <your-hf-name>/Virgo-1.0-Angkor-Hearing
    python speech/finetune_stt.py --extra my_recordings/ --steps 3000

Base: Whisper large-v3-turbo, trained with LoRA so it fits a free 16 GB T4 (Kaggle), then merged into
a normal Whisper model. Data:
  - OpenSLR 42: Google's Khmer speech recordings with transcripts (CC BY-SA 4.0)
  - Google FLEURS km_kh (CC BY 4.0), when the installed `datasets` can still load it
  - your own recordings (--extra): .wav files + metadata.csv of `file_name,sentence`
It prints the Khmer character error rate (CER) before and after training, on clips it never trained
on. The result goes to speech/out/virgo-1.0-stt, which virgo_speech.py uses automatically.

One model for every language (--multilingual): Khmer plus English and other languages from Google
FLEURS (CC BY 4.0), each clip with its own language tag, so Whisper learns Khmer without forgetting the
rest or how to tell languages apart. Start it from openai/whisper-large-v3-turbo. It's uploaded only
when Khmer is at least as good as the current hearing and the other languages stay within 15% of the
base model; the server then uses it alone (no second Whisper).

    python speech/finetune_stt.py --multilingual --base openai/whisper-large-v3-turbo --push virgoai/Angkor-1.0-STT
"""
import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # one GPU: on "T4 x2" the Trainer would split batches across both

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
MAX_LABEL_TOKENS = 440  # Whisper's decoder takes at most 448 tokens, including its start tokens
RATE = 16000


def read_extra(folder):
    with open(os.path.join(folder, "metadata.csv"), encoding="utf-8") as f:
        return [(os.path.join(folder, r["file_name"]), r["sentence"].strip()) for r in csv.DictReader(f) if r["sentence"].strip()]


def fleurs(work):
    """FLEURS Khmer, saved as wav files (skipped if this `datasets` version can't load it)."""
    import soundfile as sf

    folder = os.path.join(work, "fleurs")
    index = os.path.join(folder, "metadata.csv")
    if os.path.exists(index) and sum(1 for _ in open(index, encoding="utf-8")) <= 1:
        os.remove(index)  # left empty by a run that stopped
    if not os.path.exists(index):
        try:
            from datasets import Audio, load_dataset

            try:
                rows = load_dataset("google/fleurs", "km_kh", split="train")
            except Exception:
                rows = load_dataset("google/fleurs", "km_kh", split="train", trust_remote_code=True)
            # The audio is read here with soundfile, not by `datasets` (newer versions need torchcodec).
            rows = rows.cast_column("audio", Audio(decode=False))
        except Exception as err:
            print("FLEURS skipped (can't load it with this datasets version):", type(err).__name__)
            return []
        import io

        os.makedirs(folder, exist_ok=True)
        kept = 0
        with open(index + ".part", "w", encoding="utf-8", newline="") as f:
            out = csv.writer(f)
            out.writerow(["file_name", "sentence"])
            for i, row in enumerate(rows):
                audio = row["audio"]
                try:
                    source = io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"]
                    wave, rate = sf.read(source, dtype="float32")
                except Exception:
                    continue
                name = f"{i:05d}.wav"
                sf.write(os.path.join(folder, name), wave, rate)
                out.writerow([name, row["transcription"]])
                kept += 1
        os.replace(index + ".part", index)
        print(f"✅ FLEURS Khmer: {kept} clips")
    return read_extra(folder)


FLEURS_CONFIGS = {"en": "en_us", "zh": "cmn_hans_cn", "fr": "fr_fr", "ja": "ja_jp", "ko": "ko_kr", "th": "th_th",
                  "vi": "vi_vn", "de": "de_de", "es": "es_419", "ru": "ru_ru", "id": "id_id", "lo": "lo_la"}


def other_languages(work, spec):
    """Clips in other languages from Google FLEURS, as (path, text, language code). spec: "en:2000,zh:300"."""
    import io

    import soundfile as sf

    rows = []
    for part in filter(None, (x.strip() for x in spec.split(","))):
        code, _, count = part.partition(":")
        count, config = int(count or 300), FLEURS_CONFIGS.get(code)
        if not config:
            print(f"{code}: no FLEURS set for it, skipped")
            continue
        folder = os.path.join(work, "other", code)
        index = os.path.join(folder, "metadata.csv")
        if not os.path.exists(index):
            data = None
            for kwargs in ({"revision": "refs/convert/parquet"}, {}, {"trust_remote_code": True}):
                try:
                    from datasets import Audio, load_dataset

                    data = load_dataset("google/fleurs", config, split="train", streaming=True, **kwargs)
                    data = data.cast_column("audio", Audio(decode=False))
                    next(iter(data))
                    break
                except Exception as err:
                    data, last = None, err
            if data is None:
                print(f"FLEURS {config} skipped: {type(last).__name__}: {str(last)[:150]}")
                continue
            os.makedirs(folder, exist_ok=True)
            kept = 0
            with open(index + ".part", "w", encoding="utf-8", newline="") as f:
                out = csv.writer(f)
                out.writerow(["file_name", "sentence"])
                for i, row in enumerate(data):
                    if kept >= count:
                        break
                    text = (row.get("raw_transcription") or row.get("transcription") or "").strip()
                    try:
                        audio = row["audio"]
                        wave, rate = sf.read(io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"], dtype="float32")
                    except Exception:
                        continue
                    if not text:
                        continue
                    name = f"{i:05d}.wav"
                    sf.write(os.path.join(folder, name), wave, rate)
                    out.writerow([name, text])
                    kept += 1
            os.replace(index + ".part", index)
            print(f"✅ FLEURS {config}: {kept} clips")
        rows += [(path, text, code) for path, text in read_extra(folder)]
    return rows


TEXT_COLUMNS = ["sentence", "transcription", "transcript", "text", "normalized_text", "raw_transcription", "label"]


def hf_dataset(work, spec, limit=20000):
    """Any Hugging Face speech dataset ("name" or "name:split"): its audio and transcript columns are
    found automatically. Audio is saved as it is (no decoding here), so any format librosa reads works."""
    from datasets import Audio, load_dataset

    name, _, split = spec.partition(":")
    folder = os.path.join(work, "hf", name.replace("/", "__") + (f"__{split}" if split else ""))
    index = os.path.join(folder, "metadata.csv")
    if not os.path.exists(index):
        try:
            rows = load_dataset(name, split=split or "train")
        except Exception as err:
            print(f"{spec} skipped: {type(err).__name__}: {str(err)[:200]}")
            return []
        audio = next((c for c, f in rows.features.items() if isinstance(f, Audio)), None)
        text = next((c for c in TEXT_COLUMNS if c in rows.column_names), None)
        if not audio or not text:
            print(f"{spec} skipped: no audio or transcript column (columns: {rows.column_names})")
            return []
        rows = rows.cast_column(audio, Audio(decode=False)).select_columns([audio, text])
        os.makedirs(folder, exist_ok=True)
        with open(index, "w", encoding="utf-8", newline="") as f:
            out = csv.writer(f)
            out.writerow(["file_name", "sentence"])
            for i, row in enumerate(rows):
                if i >= limit:
                    break
                clip, sentence = row[audio], str(row[text] or "").strip()
                data = clip.get("bytes") or (open(clip["path"], "rb").read() if clip.get("path") and os.path.exists(clip["path"]) else None)
                if not data or not sentence:
                    continue
                ext = os.path.splitext(clip.get("path") or "")[1] or ".wav"
                file_name = f"{i:06d}{ext}"
                with open(os.path.join(folder, file_name), "wb") as w:
                    w.write(data)
                out.writerow([file_name, sentence])
        print(f"✅ {spec}: saved {sum(1 for _ in open(index, encoding='utf-8')) - 1} clips")
    return read_extra(folder)


def cer(ref, hyp):
    """Character error rate, ignoring spaces (Khmer doesn't put spaces between words)."""
    a, b = ref.replace(" ", ""), hyp.replace(" ", "")
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(1, len(a))


class Clips(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import librosa

        path, text, *lang = self.rows[i]
        audio, _ = librosa.load(path, sr=RATE, mono=True)
        return {"audio": audio[: 30 * RATE], "text": text, "lang": lang[0] if lang else "khmer"}


@dataclass
class Collator:
    processor: object

    def __call__(self, batch):
        features = self.processor.feature_extractor([b["audio"] for b in batch], sampling_rate=RATE, return_tensors="pt")
        # Each clip's own language tag (Khmer, English...), so Whisper keeps telling languages apart.
        tok = self.processor.tokenizer
        start = tok.convert_tokens_to_ids("<|startoftranscript|>")
        seqs = []
        for b in batch:
            tok.set_prefix_tokens(language=b["lang"], task="transcribe")
            ids = tok(b["text"]).input_ids
            seqs.append(ids[1:] if ids and ids[0] == start else ids)  # the model adds the start token itself
        tok.set_prefix_tokens(language="khmer", task="transcribe")
        labels = torch.full((len(seqs), max(map(len, seqs))), -100, dtype=torch.long)
        for i, ids in enumerate(seqs):
            labels[i, : len(ids)] = torch.tensor(ids)
        return {"input_features": features["input_features"], "labels": labels}


@torch.inference_mode()
def score(model, processor, rows, limit=100):
    """Average Khmer CER on up to `limit` held-out clips."""
    import librosa

    model.eval()
    dtype = next(model.parameters()).dtype
    total = []
    for path, text in rows[:limit]:
        audio, _ = librosa.load(path, sr=RATE, mono=True)
        feats = processor.feature_extractor(audio[: 30 * RATE], sampling_rate=RATE, return_tensors="pt").input_features
        ids = model.generate(feats.to(model.device, dtype), language="khmer", task="transcribe", max_new_tokens=200)
        total.append(cer(text, processor.tokenizer.batch_decode(ids, skip_special_tokens=True)[0]))
    return float(np.mean(total)) if total else float("nan")


@torch.inference_mode()
@torch.inference_mode()
def detect_language(model, tokenizer, features):
    """Whisper's language guess for one clip: the most likely language token right after the start token
    (generate()'s output may leave the language tag out, depending on the transformers version)."""
    lang_to_id = getattr(model.generation_config, "lang_to_id", None) or {}
    if not lang_to_id:
        return None
    start = torch.tensor([[model.generation_config.decoder_start_token_id]], device=model.device)
    logits = model(input_features=features, decoder_input_ids=start).logits[0, -1]
    codes, ids = zip(*lang_to_id.items())
    return codes[int(torch.argmax(logits[list(ids)]))].strip("<|>")


@torch.inference_mode()
def score_detect(model, processor, rows, limit=60):
    """With the language left to the model (as in live voice): (average CER, share of clips whose language
    it got right) on up to `limit` held-out clips of (path, text, language)."""
    import librosa

    model.eval()
    dtype = next(model.parameters()).dtype
    errors, right = [], 0
    rows = rows[:limit]
    for path, text, lang in rows:
        audio, _ = librosa.load(path, sr=RATE, mono=True)
        feats = processor.feature_extractor(audio[: 30 * RATE], sampling_rate=RATE, return_tensors="pt").input_features.to(model.device, dtype)
        guess = detect_language(model, processor.tokenizer, feats)
        right += guess == {"khmer": "km"}.get(lang, lang)
        ids = model.generate(feats, task="transcribe", max_new_tokens=200, **({"language": guess} if guess else {}))
        errors.append(cer(text.lower(), processor.tokenizer.decode(ids[0], skip_special_tokens=True).lower()))
    return (float(np.mean(errors)) if errors else float("nan")), (right / len(rows) if rows else float("nan"))


def main():
    p = argparse.ArgumentParser(description="Fine-tune Virgo's Khmer hearing")
    p.add_argument("--base", default="openai/whisper-large-v3-turbo")
    p.add_argument("--out", default="speech/out/virgo-1.0-stt")
    p.add_argument("--work", default="speech/work/stt")
    p.add_argument("--extra", help="your own recordings: .wav files + metadata.csv (file_name,sentence)")
    p.add_argument("--no-slr42", action="store_true")
    p.add_argument("--no-fleurs", action="store_true")
    p.add_argument("--hf", action="append", default=[], metavar="NAME[:SPLIT]",
                   help="also train on a Hugging Face Khmer speech dataset (repeat for more); check its license first")
    p.add_argument("--hf-max", type=int, default=20000, help="at most this many clips from each --hf dataset")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4, help="LoRA learning rate (use ~1e-5 with --full)")
    p.add_argument("--full", action="store_true", help="train every weight instead of LoRA (needs more GPU memory)")
    p.add_argument("--push", help="Hugging Face repo to save to, e.g. you/Virgo-1.0-Angkor-Hearing")
    p.add_argument("--score-only", metavar="FOLDER", help="don't train: score (and upload, if it passes) a model already trained into FOLDER")
    p.add_argument("--multilingual", action="store_true", help="one model for every language: also train on other languages (--keep)")
    p.add_argument("--keep", default="en:2500,zh:300,th:300,vi:300,fr:200,ja:200,ko:200",
                   help="with --multilingual: languages (and clips) to keep, from Google FLEURS")
    args = p.parse_args()
    if not args.multilingual:  # a hearing already trained on every language stays that way (else it forgets them)
        sys.path.insert(0, os.path.dirname(__file__))
        from virgo_speech_marker import multilingual

        if multilingual(args.base):
            print("ℹ️ The base hears every language: training stays multilingual (--multilingual)")
            args.multilingual = True

    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor

    rows = []
    if not args.no_slr42:
        from train_voice import download_slr42

        rows += [(path, text) for _, path, text in download_slr42(args.work)]
    if not args.no_fleurs:
        rows += fleurs(args.work)
    for spec in args.hf:
        rows += hf_dataset(args.work, spec, args.hf_max)
    if args.extra:
        rows += read_extra(args.extra)
    if len(rows) < 20:
        raise SystemExit("❌ Not enough training clips.")
    random.Random(42).shuffle(rows)
    held = max(10, min(200, len(rows) // 20))
    evals, train = rows[:held], rows[held:]
    others, other_evals = [], []
    if args.multilingual:
        others = other_languages(args.work, args.keep)
        if not any(lang == "en" for *_, lang in others):
            raise SystemExit("❌ --multilingual needs English clips (FLEURS en_us couldn't be loaded): without them the "
                             "model forgets English again. Try another `datasets` version, or run without --multilingual.")
        random.Random(7).shuffle(others)
        other_evals, others = others[:120], others[120:]
    print(f"✅ {len(train)} Khmer + {len(others)} other-language training clips, {len(evals)} Khmer held out for scoring")

    gpu = torch.cuda.is_available()
    processor = WhisperProcessor.from_pretrained(args.base, language="khmer", task="transcribe")
    # Khmer text uses many tokens: drop clips whose transcript is too long for Whisper's decoder.
    fits = lambda text: len(processor.tokenizer(text).input_ids) <= MAX_LABEL_TOKENS
    kept = [r for r in train if fits(r[1])]
    evals = [r for r in evals if fits(r[1])]
    if len(kept) < len(train):
        print(f"Skipped {len(train) - len(kept)} clips with transcripts too long for Whisper")
    train = kept + [r for r in others if fits(r[1])]
    random.Random(1).shuffle(train)
    model = WhisperForConditionalGeneration.from_pretrained(args.base, dtype=torch.float32)
    model.generation_config.forced_decoder_ids = None
    if gpu:
        model.to("cuda")
    before = score(model, processor, evals)
    print(f"Khmer CER before: {before:.1%}")
    if other_evals:
        other_before, lid_before = score_detect(model, processor, other_evals)
        print(f"Other languages before: CER {other_before:.1%}, language right {lid_before:.0%}")
    # Starting from another Whisper (e.g. a community Khmer fine-tune): the upload must also beat the
    # hearing Virgo has now, scored on the same clips.
    current = float("inf")
    if args.push and args.push != args.base:
        try:
            from huggingface_hub import HfApi

            if HfApi().file_exists(args.push, "config.json"):
                old_model = WhisperForConditionalGeneration.from_pretrained(args.push, dtype=torch.float32)
                old_model.generation_config.forced_decoder_ids = None
                current = score(old_model.to("cuda") if gpu else old_model, processor, evals)
                print(f"Khmer CER of the current {args.push}: {current:.1%}")
                del old_model
                if gpu:
                    torch.cuda.empty_cache()
        except Exception as err:
            print("Couldn't score the current hearing:", err)

    if args.score_only:  # a model trained earlier (e.g. stopped by a scoring bug): score it as "after"
        del model
        model = WhisperForConditionalGeneration.from_pretrained(args.score_only, dtype=torch.float32)
        model.generation_config.forced_decoder_ids = None
        model = model.to("cuda") if gpu else model
        args.full = True  # nothing to merge
    elif not args.full:
        from peft import LoraConfig, get_peft_model

        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
                                                 target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]))
        model.print_trainable_parameters()
    if not args.score_only:
        model.train()
    trainer = None if args.score_only else Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(
            output_dir=os.path.join(args.work, "run"), per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=max(1, 16 // args.batch), learning_rate=args.lr, warmup_steps=min(200, args.steps // 10),
            max_steps=args.steps, fp16=gpu, gradient_checkpointing=gpu,
            gradient_checkpointing_kwargs={"use_reentrant": False} if gpu else None,
            logging_steps=25, save_strategy="no", report_to=[], remove_unused_columns=False,
            label_names=["labels"], dataloader_num_workers=2 if gpu else 0,
        ),
        train_dataset=Clips(train),
        data_collator=Collator(processor),
    )
    if trainer:
        trainer.train()

    if not args.full:
        model = model.merge_and_unload()
    after = score(model, processor, evals)
    print(f"Khmer CER after: {after:.1%} (before: {before:.1%})")
    keeps_others = True
    if other_evals:
        other_after, lid_after = score_detect(model, processor, other_evals)
        _, khmer_lid = score_detect(model, processor, [(p, t, "khmer") for p, t in evals])
        print(f"Other languages after: CER {other_after:.1%} (before {other_before:.1%}), language right {lid_after:.0%} "
              f"(before {lid_before:.0%}); Khmer recognized as Khmer: {khmer_lid:.0%}")
        keeps_others = other_after <= other_before * 1.15 + 0.01 and lid_after >= lid_before - 0.05 and khmer_lid >= 0.85
        if not keeps_others:
            print("⚠️ Other languages or language detection got too much worse: not uploaded (more --keep clips may help).")
    model.generation_config.forced_decoder_ids = None
    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    if args.multilingual:  # the server uses this one model for every language
        import json

        with open(os.path.join(args.out, "virgo_hearing.json"), "w", encoding="utf-8") as f:
            json.dump({"multilingual": True, "languages": ["km"] + sorted({lang for *_, lang in others})}, f)
    with open(os.path.join(args.out, "score.txt"), "w", encoding="utf-8") as f:
        f.write(f"base {args.base}\ncurrent {current:.4f}\nclips {len(train)} train / {len(evals)} held out\nkhmer CER before {before:.4f}\nkhmer CER after {after:.4f}\n")
    print("✅ Saved Virgo's Khmer hearing to", args.out)
    if after > before:
        print("⚠️ It got worse on the held-out clips: don't use this one (try fewer --steps or a lower --lr).")
    if not keeps_others:
        args.push = None
    if args.push and after <= before and after > current:
        print(f"⚠️ Not uploaded: the current {args.push} is still better ({current:.1%} vs {after:.1%}).")
    if args.push and after <= before and after <= current:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push, private=True, exist_ok=True)
        api.upload_folder(folder_path=args.out, repo_id=args.push, commit_message=f"Khmer CER {before:.1%} → {after:.1%}")
        print("☁️ Uploaded to", args.push)


if __name__ == "__main__":
    main()
