"""Train Virgo 1.0 chat: a LoRA adapter on an open model (Gemma 3 4B by default).

    pip install -r requirements.txt
    python chat/train.py                                  # Gemma 3 4B, uses chat/data/*.jsonl
    python chat/train.py --base google/gemma-3-1b-it      # lighter and faster, less smart
    python chat/train.py --base google/gemma-3-12b-it     # smartest; 4 bits on a free T4 (Kaggle), ~1–3 h

The 4B model is loaded in 4 bits (QLoRA) on a GPU, so it trains on a free Google Colab T4 in well
under an hour for a few thousand examples. On a CPU, use the 1B model. The adapter is saved to chat/out/virgo-1.0-chat-lora; add --merge to also save
a full merged model (for ONNX/browser export or servers without LoRA support).

Data: one JSON object per line, {"messages": [{"role": "system"|"user"|"assistant", "content": ...}]}.
Check it with `python scripts/check_data.py` before training.
"""
import argparse
import glob
import json
import math
import re

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

DEFAULT_BASE = "google/gemma-3-4b-it"


def load_base(name, four_bit):
    """The base model; Gemma 3 4B and up are image+text models, so fall back to their full class."""
    kwargs = {}
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if four_bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=kwargs["torch_dtype"],
            )
            kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.float32
    try:
        return AutoModelForCausalLM.from_pretrained(name, **kwargs)
    except ValueError:
        from transformers import Gemma3ForConditionalGeneration

        return Gemma3ForConditionalGeneration.from_pretrained(name, **kwargs)


def load_examples(pattern):
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            rows += [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise SystemExit(f"No training data found at {pattern}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--data", default="chat/data/*.jsonl")
    p.add_argument("--out", default="chat/out/virgo-1.0-chat-lora")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--merge", action="store_true", help="also save a merged full model")
    p.add_argument("--full-precision", action="store_true", help="don't load the base in 4 bits (needs more GPU memory)")
    p.add_argument("--batch", type=int, help="examples per step (default: 4, or 1 for 12B and up)")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.pad_token = tok.pad_token or tok.eos_token
    gpu = torch.cuda.is_available()
    four_bit = gpu and not args.full_precision
    model = load_base(args.base, four_bit)
    if four_bit:
        # Not peft's prepare_model_for_kbit_training: it turns every unquantized weight into float32,
        # and Gemma 3's 262k-word embedding alone then takes 4 GB (12B) to 5.6 GB (27B) of GPU memory.
        # Gradient checkpointing (activations recomputed instead of kept) is what's needed.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.config.use_cache = False

    # Gemma's chat template has no system role: fold the system message into the first user turn.
    def to_text(example):
        msgs = example["messages"]
        if msgs and msgs[0]["role"] == "system" and "gemma" in args.base.lower():
            system, rest = msgs[0]["content"], msgs[1:]
            rest = [{**rest[0], "content": f"{system}\n\n{rest[0]['content']}"}] + rest[1:]
            msgs = rest
        return {"text": tok.apply_chat_template(msgs, tokenize=False)}

    ds = Dataset.from_list(load_examples(args.data)).map(to_text, remove_columns=["messages"])
    ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=args.max_len), batched=True, remove_columns=["text"])

    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()
    # 12B and up: one example at a time (same effective batch of 16), 8-bit optimizer, so it fits a 16 GB T4.
    big = max([int(n) for n in re.findall(r"(\d+)b", args.base.split("/")[-1].lower())] or [0]) >= 12
    batch = args.batch or (1 if big else 4)

    Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
            per_device_train_batch_size=batch, gradient_accumulation_steps=max(1, 16 // batch),
            optim="paged_adamw_8bit" if four_bit and big else "adamw_torch",
            # warmup_steps works on every transformers version (warmup_ratio was removed in newer ones)
            warmup_steps=max(1, int(0.05 * math.ceil(len(ds) / 16) * args.epochs)),
            logging_steps=10, save_strategy="no", report_to=[],
            gradient_checkpointing=four_bit, gradient_checkpointing_kwargs={"use_reentrant": False} if four_bit else None,
            bf16=gpu and torch.cuda.is_bf16_supported(), fp16=gpu and not torch.cuda.is_bf16_supported(),
        ),
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    ).train()

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"Saved the Virgo 1.0 chat adapter to {args.out}")
    if args.merge:
        if four_bit:  # merge into a full-precision copy of the base, not the 4-bit one
            from peft import PeftModel

            model = PeftModel.from_pretrained(load_base(args.base, False), args.out)
        merged = model.merge_and_unload()
        merged.save_pretrained(f"{args.out}-merged")
        tok.save_pretrained(f"{args.out}-merged")
        print(f"Saved the merged model to {args.out}-merged")


if __name__ == "__main__":
    main()
