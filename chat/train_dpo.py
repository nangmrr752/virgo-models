"""DPO: teaches a trained Virgo chat model the difference between a good and a bad answer to the same
question (right vs wrong number, Khmer vs the wrong language, the format asked for vs not), from the
pairs scripts/distill_smart.py saves (chat/dpo/*.jsonl).

    python chat/train_dpo.py --adapter <data>/out/Angkor-1.0-12B --out <data>/out/Angkor-1.0-12B-dpo

It starts from the model's own trained adapter (SFT) and keeps a frozen copy of it as the reference, so
the model only moves toward the better answers and doesn't forget the rest. Three epochs at 2e-5
(what worked in round 2). Pair format: {"prompt": [messages], "chosen": "...", "rejected": "..."}.
"""
import argparse
import glob
import json
import os
import sys

import torch

sys.path.append(os.path.dirname(__file__))
from train import load_base  # noqa: E402
from virgo_chat import adapter_base  # noqa: E402


def load_pairs(pattern, tok, gemma):
    rows = []
    for path in sorted(glob.glob(pattern)):
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            if not all(k in row for k in ("prompt", "chosen", "rejected")):  # e.g. student_failed.jsonl: not a pair
                continue
            msgs = row["prompt"]
            if gemma and msgs and msgs[0]["role"] == "system":  # Gemma has no system role: fold it in
                msgs = [{**msgs[1], "content": f"{msgs[0]['content']}\n\n{msgs[1]['content']}"}] + msgs[2:]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            if row["chosen"].strip() and row["rejected"].strip() and row["chosen"] != row["rejected"]:
                rows.append({"prompt": prompt, "chosen": row["chosen"], "rejected": row["rejected"]})
    if not rows:
        raise SystemExit(f"No DPO pairs found at {pattern} (make them with: bash scripts/train_local.sh distill)")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True, help="the trained (SFT) adapter to improve")
    p.add_argument("--pairs", default="chat/dpo/*.jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--beta", type=float, default=0.1, help="how far it may move from the SFT model (lower = further)")
    p.add_argument("--lr", type=float, default=2e-5)  # round 2: 5e-6 barely moved the model, 2e-5 x 3 epochs gave +1.3 points
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--max-len", type=int, default=1024)
    args = p.parse_args()

    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    base = adapter_base(args.adapter)
    tok = AutoTokenizer.from_pretrained(args.adapter)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = load_base(base, four_bit=torch.cuda.is_available())
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    # The adapter twice: "default" learns (TRL looks for that name), "reference" stays as the SFT model it is compared with.
    # (Not "train": that's a PyTorch method name, so PEFT can't use it.)
    model = PeftModel.from_pretrained(model, args.adapter, adapter_name="default", is_trainable=True)
    model.load_adapter(args.adapter, adapter_name="reference")
    model.set_adapter("default")

    rows = load_pairs(args.pairs, tok, "gemma" in base.lower())
    print(f"{len(rows)} DPO pairs", flush=True)
    config = dict(
        output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr, beta=args.beta,
        per_device_train_batch_size=1, gradient_accumulation_steps=16, lr_scheduler_type="cosine", warmup_ratio=0.05,
        max_length=args.max_len, max_prompt_length=min(512, args.max_len // 2), logging_steps=10, save_strategy="no",
        report_to=[], bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        model_adapter_name="default", ref_adapter_name="reference", optim="paged_adamw_8bit",
    )
    import inspect

    known = inspect.signature(DPOConfig.__init__).parameters
    config = {k: v for k, v in config.items() if k in known}  # options differ between trl versions
    trainer_kwargs = {"model": model, "args": DPOConfig(**config), "train_dataset": Dataset.from_list(rows)}
    trainer_params = inspect.signature(DPOTrainer.__init__).parameters
    trainer_kwargs["processing_class" if "processing_class" in trainer_params else "tokenizer"] = tok
    DPOTrainer(**trainer_kwargs).train()

    model.save_pretrained(args.out, selected_adapters=["default"])
    # peft saves a named adapter in a subfolder: move it up so the folder loads like any Virgo adapter.
    sub = os.path.join(args.out, "default")
    if os.path.isdir(sub):
        for name in os.listdir(sub):
            os.replace(os.path.join(sub, name), os.path.join(args.out, name))
        os.rmdir(sub)
    tok.save_pretrained(args.out)
    print("Saved the DPO adapter to", args.out)


if __name__ == "__main__":
    main()
