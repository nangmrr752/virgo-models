"""Train Virgo 1.0 chat: a LoRA adapter on a small open model (Gemma 3 1B by default).

    pip install -r requirements.txt
    python chat/train.py                       # uses chat/data/*.jsonl
    python chat/train.py --base google/gemma-3-1b-it --epochs 3

Runs on a free Google Colab T4 GPU in well under an hour for a few thousand examples. On a CPU it
works but is slow. The adapter is saved to chat/out/virgo-1.0-chat-lora; add --merge to also save
a full merged model (for ONNX/browser export or servers without LoRA support).

Data: one JSON object per line, {"messages": [{"role": "system"|"user"|"assistant", "content": ...}]}.
Check it with `python scripts/check_data.py` before training.
"""
import argparse
import glob
import json

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments


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
    p.add_argument("--base", default="google/gemma-3-1b-it")
    p.add_argument("--data", default="chat/data/*.jsonl")
    p.add_argument("--out", default="chat/out/virgo-1.0-chat-lora")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--merge", action="store_true", help="also save a merged full model")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.pad_token = tok.pad_token or tok.eos_token
    gpu = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16 if gpu else torch.float32)

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

    Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
            per_device_train_batch_size=4, gradient_accumulation_steps=4, warmup_ratio=0.05,
            logging_steps=10, save_strategy="no", bf16=gpu, report_to=[],
        ),
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    ).train()

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"Saved the Virgo 1.0 chat adapter to {args.out}")
    if args.merge:
        merged = model.merge_and_unload()
        merged.save_pretrained(f"{args.out}-merged")
        tok.save_pretrained(f"{args.out}-merged")
        print(f"Saved the merged model to {args.out}-merged")


if __name__ == "__main__":
    main()
