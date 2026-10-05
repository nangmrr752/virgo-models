"""Prepares Angkor-1.0's LoRA adapter for Cloudflare Workers AI, so the website runs Angkor on Cloudflare's
own GPUs (Gemma 3 12B + this adapter) with no Virgo server needed for chat.

    python scripts/cloudflare_lora.py                       # Angkor-1.0-12B from Hugging Face (main)
    python scripts/cloudflare_lora.py --adapter /opt/virgo/out/Angkor-1.0-12B --out /opt/virgo/out/angkor-cloudflare

Then upload it (once per new Angkor version) from a computer logged in to Cloudflare (npx wrangler login):

    npx wrangler ai finetune create @cf/google/gemma-3-12b-it angkor-1-0 /opt/virgo/out/angkor-cloudflare

and set the Worker variable ANGKOR_LORA=angkor-1-0 (wrangler.toml [vars] or the dashboard).

Workers AI's limits for LoRAs: rank ≤ 32 and adapter_model.safetensors ≤ 300 MB; the adapter_config.json
must say which kind of model it is ("model_type": "gemma"). This script checks the rank and size, saves the
weights in 16 bits (half the size of 32-bit) and writes the two files Workers AI takes.
"""
import argparse
import json
import os
import shutil
import sys

MAX_RANK = 32
MAX_MB = 300


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="", help="adapter folder; empty: download Angkor-1.0-12B from Hugging Face")
    p.add_argument("--repo", default="", help="Hugging Face repo to download (default: Angkor-1.0-12B's)")
    p.add_argument("--revision", default="main")
    p.add_argument("--out", default="out/angkor-cloudflare")
    args = p.parse_args()

    folder = args.adapter
    if not folder:
        from huggingface_hub import snapshot_download

        repo = args.repo
        if not repo:
            sys.path.append(os.path.dirname(__file__))
            from names import resolve  # noqa: E402

            repo = resolve("Angkor-1.0-12B")
        folder = snapshot_download(repo, revision=args.revision, allow_patterns=["adapter_config.json", "adapter_model.safetensors"])
        print("Downloaded", repo, f"({args.revision})")

    config = json.load(open(os.path.join(folder, "adapter_config.json")))
    base = config.get("base_model_name_or_path", "")
    if "gemma-3-12b" not in base.lower():
        raise SystemExit(f"❌ This adapter is for {base}; Workers AI runs it only on Gemma 3 12B (@cf/google/gemma-3-12b-it).")
    rank = int(config.get("r", 0))
    if rank > MAX_RANK:
        raise SystemExit(f"❌ Rank {rank}: Workers AI takes LoRAs up to rank {MAX_RANK}. Retrain with: python chat/train.py --rank {MAX_RANK} ...")

    from safetensors.torch import load_file, save_file

    weights = load_file(os.path.join(folder, "adapter_model.safetensors"))
    weights = {k: v.half().contiguous() for k, v in weights.items()}
    os.makedirs(args.out, exist_ok=True)
    out_weights = os.path.join(args.out, "adapter_model.safetensors")
    save_file(weights, out_weights, metadata={"format": "pt"})
    size_mb = os.path.getsize(out_weights) / 1e6
    if size_mb > MAX_MB:
        shutil.rmtree(args.out)
        raise SystemExit(f"❌ {size_mb:.0f} MB: Workers AI takes LoRA files up to {MAX_MB} MB. Retrain with a lower --rank.")

    config["model_type"] = "gemma"
    json.dump(config, open(os.path.join(args.out, "adapter_config.json"), "w"), indent=2)
    print(f"✅ Ready for Workers AI: {args.out} (rank {rank}, {size_mb:.0f} MB, {len(weights)} tensors)")
    print("Upload it (from a computer logged in with npx wrangler login):")
    print(f"  npx wrangler ai finetune create @cf/google/gemma-3-12b-it angkor-1-0 {os.path.abspath(args.out)}")
    print("Then set the website's Worker variable ANGKOR_LORA=angkor-1-0")


if __name__ == "__main__":
    main()
