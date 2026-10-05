"""Keeps Angkor-2.0's model file under Virgo's own Hugging Face account: copies the GGUF the llama
service runs (Gemma 4 26B A4B, Unsloth's 4-bit UD-Q4_K_XL) to a private repo with a model card, so Virgo
keeps working even if the original repo changes or disappears.

    python scripts/backup_gguf.py                                   # → virgoai/Angkor-2.0 (private)
    python scripts/backup_gguf.py --source unsloth/gemma-4-26B-A4B-it-GGUF --file gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf

Then .env:  VIRGO_LLAMA_MODEL=virgoai/Angkor-2.0:UD-Q4_K_XL   (llama.cpp loads it from your repo).
The file is already in the Hugging Face cache, so only the upload takes time (~16 GB).
Gemma 4 is Apache 2.0: copying is allowed with the license and attribution (in the model card).
"""
import argparse
import os
import re
import sys

CARD = """---
license: apache-2.0
base_model: google/gemma-4-26b-a4b-it
tags: [gguf, gemma-4, khmer, virgo, ksn]
language: [km, en, th, zh, ja, ko, fr, es, vi, de]
---

# {name}

Virgo's main chat model, made by KSN: **Google Gemma 4 26B A4B** (Mixture of Experts, about 4B active per word),
4-bit GGUF ({quant}) for llama.cpp, served with Virgo's own instructions on KSN's Virgo server.

| | |
|---|---|
| Base model | [google/gemma-4-26b-a4b-it](https://huggingface.co/google/gemma-4-26b-a4b-it) (Apache 2.0) |
| This file | `{file}`, copied unchanged from [{source}](https://huggingface.co/{source}) |
| Training | none yet (Virgo's instructions only) |
| Virgo big hard test | 97.4% (326 questions; Angkor-1.0: 94.8%) |

Licensed under the Apache License 2.0, like Gemma 4. Credit: Google DeepMind (Gemma 4), Unsloth (GGUF quantization).
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="unsloth/gemma-4-26B-A4B-it-GGUF")
    p.add_argument("--file", default="gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf")
    p.add_argument("--name", default="Angkor-2.0")
    p.add_argument("--repo", default="", help="default: <VIRGO_HF_ORG or your account>/<name>")
    p.add_argument("--public", action="store_true", help="make the repo public (default: private)")
    args = p.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    org = os.environ.get("VIRGO_HF_ORG") or api.whoami()["name"]
    repo = args.repo or f"{org}/{args.name}"
    found = re.search(r"((?:UD-)?I?Q\d\w*)\.gguf$", args.file)
    quant = found.group(1) if found else args.file.removesuffix(".gguf")
    print(f"Getting {args.source}/{args.file} (from the cache when it's there)…", flush=True)
    path = hf_hub_download(args.source, args.file)
    print(f"{path}: {os.path.getsize(path) / 1e9:.1f} GB", flush=True)

    api.create_repo(repo, private=not args.public, exist_ok=True)
    if api.file_exists(repo, args.file):
        print(f"{repo} already has {args.file}: not uploading it again")
    else:
        print(f"Uploading to {repo} ({'public' if args.public else 'private'})… this takes a while", flush=True)
        api.upload_file(path_or_fileobj=path, path_in_repo=args.file, repo_id=repo, commit_message=f"{args.name}: {args.file}")
    card = CARD.format(name=args.name, quant=quant, file=args.file, source=args.source)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo, commit_message="Model card")
    print(f"✅ https://huggingface.co/{repo}")
    print(f"Now in .env:  VIRGO_LLAMA_MODEL={repo}:{quant}   then: docker compose up -d llama")


if __name__ == "__main__":
    sys.exit(main())
