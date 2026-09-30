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
    if not os.path.exists(index):
        try:
            from datasets import load_dataset

            rows = load_dataset("google/fleurs", "km_kh", split="train", trust_remote_code=True)
        except Exception as err:
            print("FLEURS skipped (can't load it with this datasets version):", type(err).__name__)
            return []
        os.makedirs(folder, exist_ok=True)
        with open(index, "w", encoding="utf-8", newline="") as f:
            out = csv.writer(f)
            out.writerow(["file_name", "sentence"])
            for i, row in enumerate(rows):
                name = f"{i:05d}.wav"
                sf.write(os.path.join(folder, name), row["audio"]["array"], row["audio"]["sampling_rate"])
                out.writerow([name, row["transcription"]])
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

        path, text = self.rows[i]
        audio, _ = librosa.load(path, sr=RATE, mono=True)
        return {"audio": audio[: 30 * RATE], "text": text}


@dataclass
class Collator:
    processor: object

    def __call__(self, batch):
        features = self.processor.feature_extractor([b["audio"] for b in batch], sampling_rate=RATE, return_tensors="pt")
        labels = self.processor.tokenizer([b["text"] for b in batch], padding=True, return_tensors="pt")
        ids = labels["input_ids"].masked_fill(labels["attention_mask"].ne(1), -100)
        # The model adds the start token itself: drop it from the labels when the tokenizer added it.
        start = self.processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        if (ids[:, 0] == start).all():
            ids = ids[:, 1:]
        return {"input_features": features["input_features"], "labels": ids}


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


def main():
    p = argparse.ArgumentParser(description="Fine-tune Virgo's Khmer hearing")
    p.add_argument("--base", default="openai/whisper-large-v3-turbo")
    p.add_argument("--out", default="speech/out/virgo-1.0-stt")
    p.add_argument("--work", default="speech/work/stt")
    p.add_argument("--extra", help="your own recordings: .wav files + metadata.csv (file_name,sentence)")
    p.add_argument("--no-slr42", action="store_true")
    p.add_argument("--no-fleurs", action="store_true")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4, help="LoRA learning rate (use ~1e-5 with --full)")
    p.add_argument("--full", action="store_true", help="train every weight instead of LoRA (needs more GPU memory)")
    p.add_argument("--push", help="Hugging Face repo to save to, e.g. you/Virgo-1.0-Angkor-Hearing")
    args = p.parse_args()

    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor

    rows = []
    if not args.no_slr42:
        from train_voice import download_slr42

        rows += [(path, text) for _, path, text in download_slr42(args.work)]
    if not args.no_fleurs:
        rows += fleurs(args.work)
    if args.extra:
        rows += read_extra(args.extra)
    if len(rows) < 20:
        raise SystemExit("❌ Not enough training clips.")
    random.Random(42).shuffle(rows)
    held = max(10, min(200, len(rows) // 20))
    evals, train = rows[:held], rows[held:]
    print(f"✅ {len(train)} training clips, {len(evals)} held out for scoring")

    gpu = torch.cuda.is_available()
    processor = WhisperProcessor.from_pretrained(args.base, language="khmer", task="transcribe")
    # Khmer text uses many tokens: drop clips whose transcript is too long for Whisper's decoder.
    fits = lambda text: len(processor.tokenizer(text).input_ids) <= MAX_LABEL_TOKENS
    kept = [r for r in train if fits(r[1])]
    evals = [r for r in evals if fits(r[1])]
    if len(kept) < len(train):
        print(f"Skipped {len(train) - len(kept)} clips with transcripts too long for Whisper")
    train = kept
    model = WhisperForConditionalGeneration.from_pretrained(args.base, dtype=torch.float32)
    model.generation_config.forced_decoder_ids = None
    if gpu:
        model.to("cuda")
    before = score(model, processor, evals)
    print(f"Khmer CER before: {before:.1%}")

    if not args.full:
        from peft import LoraConfig, get_peft_model

        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
                                                 target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]))
        model.print_trainable_parameters()
    model.train()
    Seq2SeqTrainer(
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
    ).train()

    if not args.full:
        model = model.merge_and_unload()
    after = score(model, processor, evals)
    print(f"Khmer CER after: {after:.1%} (before: {before:.1%})")
    model.generation_config.forced_decoder_ids = None
    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    with open(os.path.join(args.out, "score.txt"), "w", encoding="utf-8") as f:
        f.write(f"base {args.base}\nclips {len(train)} train / {len(evals)} held out\nkhmer CER before {before:.4f}\nkhmer CER after {after:.4f}\n")
    print("✅ Saved Virgo's Khmer hearing to", args.out)
    if after > before:
        print("⚠️ It got worse on the held-out clips: don't use this one (try fewer --steps or a lower --lr).")
    if args.push and after <= before:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push, private=True, exist_ok=True)
        api.upload_folder(folder_path=args.out, repo_id=args.push, commit_message=f"Khmer CER {before:.1%} → {after:.1%}")
        print("☁️ Uploaded to", args.push)


if __name__ == "__main__":
    main()
