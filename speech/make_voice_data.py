"""Training clips for Virgo's live voice (a VoxCPM2 LoRA, used by vox_voice.py), run in the voice environment.

By default the clips are Virgo's own voice: VoxCPM2 reads Khmer sentences from Virgo's chat data in
the voice of voice.wav (Virgo-1.0-Angkor-Voice). Training on them makes the voice steadier: the same
voice every sentence, fewer odd sounds. With --recordings, your own recordings are used instead
(a folder of .wav files and metadata.csv `file_name,sentence`): the voice then becomes that speaker's,
so record only a voice you have the rights to.

    python speech/make_voice_data.py --repo you/Virgo-1.0-Angkor-Voice --out <work> --count 600
Writes <work>/clips/*.wav and <work>/train.jsonl ({"audio", "text"}, VoxCPM's training format).
"""
import argparse
import csv
import glob
import json
import os
import random
import re
import sys

sys.path.append(os.path.dirname(__file__))
from vox_voice import KHMER, load_model, voice_settings  # noqa: E402

SENTENCE = re.compile(r"(?<=[។៕!?.])\s*")


def khmer_sentences(pattern):
    """Short Khmer sentences from the assistant replies in Virgo's chat data."""
    found = set()
    for path in glob.glob(pattern):
        for line in open(path, encoding="utf-8"):
            try:
                msgs = json.loads(line)["messages"]
            except (ValueError, KeyError):
                continue
            for m in msgs:
                if m.get("role") != "assistant":
                    continue
                for s in SENTENCE.split(re.sub(r"[*#`_>|\[\]]", "", m["content"])):
                    s = " ".join(s.replace('"', " ").replace("“", " ").replace("”", " ").split()).strip("()=-–: ")
                    if s and KHMER.match(s) and "(" not in s and 12 <= len(s) <= 140 and len(KHMER.findall(s)) >= 0.6 * len(s.replace(" ", "")):
                        found.add(s)
    return sorted(found)


DESIGN_TEXT = "សួស្តី ខ្ញុំជា Virgo ជាជំនួយការ AI របស់ KSN។ ខ្ញុំរីករាយនឹងជួយអ្នក។"


def design_voice(ref, repo, description):
    """A new voice from a description: VoxCPM2 speaks one Khmer sentence in it, and that clip becomes the
    voice sample (voice.wav + voice.json), saved to `repo`, so every later sentence sounds the same."""
    import soundfile as sf
    from huggingface_hub import HfApi

    print(f"Making a new voice: {description}", flush=True)
    model = load_model(ref)
    best = None
    for seed in range(4):  # a few tries; keep the one with the most natural length
        import torch

        torch.manual_seed(seed)
        wave = model.generate(text=f"({description}){DESIGN_TEXT}", cfg_value=2.0, inference_timesteps=10)
        seconds = len(wave) / model.tts_model.sample_rate
        if 3 <= seconds <= 12 and (best is None or abs(seconds - 6) < abs(best[1] - 6)):
            best = (wave, seconds)
    if best is None:
        sys.exit("❌ Couldn't make a clean voice sample from that description: try other words.")
    sf.write(os.path.join(ref, "voice.wav"), best[0], model.tts_model.sample_rate)
    with open(os.path.join(ref, "voice.json"), "w", encoding="utf-8") as f:
        json.dump({"text": DESIGN_TEXT, "description": description}, f, ensure_ascii=False)
    api = HfApi()
    api.create_repo(repo, private=True, exist_ok=True)
    for name in ("voice.wav", "voice.json"):
        api.upload_file(path_or_fileobj=os.path.join(ref, name), path_in_repo=name, repo_id=repo, commit_message="Voice sample")
    print(f"✅ New voice sample saved to {repo} (listen: {os.path.join(ref, 'voice.wav')})", flush=True)
    del model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", help="Hugging Face repo with voice.wav (Virgo-1.0-Angkor-Voice)")
    p.add_argument("--out", required=True)
    p.add_argument("--count", type=int, default=600)
    p.add_argument("--data", default="chat/data/*.jsonl")
    p.add_argument("--design", help="no voice.wav in --repo yet: make one from this description (a new voice, e.g. for Virgo-1.0-Bayon)")
    p.add_argument("--recordings", help="your own recordings: .wav files + metadata.csv (file_name,sentence)")
    args = p.parse_args()
    os.makedirs(os.path.join(args.out, "clips"), exist_ok=True)
    manifest = os.path.join(args.out, "train.jsonl")

    if args.recordings:
        rows = list(csv.DictReader(open(os.path.join(args.recordings, "metadata.csv"), encoding="utf-8")))
        with open(manifest, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"audio": os.path.abspath(os.path.join(args.recordings, r["file_name"])), "text": r["sentence"].strip()}, ensure_ascii=False) + "\n")
        print(f"✅ {len(rows)} of your recordings")
        return

    import soundfile as sf
    from huggingface_hub import snapshot_download

    ref = os.path.join(args.out, "reference")  # only the voice sample: no old LoRA while making clips
    try:
        snapshot_download(args.repo, local_dir=ref, allow_patterns=["voice.wav", "voice.json"])
    except Exception:  # a new voice: its repo doesn't exist yet
        os.makedirs(ref, exist_ok=True)
    if not os.path.exists(os.path.join(ref, "voice.wav")):
        if not args.design:
            sys.exit("❌ No voice.wav in " + args.repo + ": make Virgo's voice first (speech/design_voice.ipynb).")
        design_voice(ref, args.repo, args.design)
    sentences = khmer_sentences(args.data)
    random.Random(7).shuffle(sentences)
    sentences = sentences[: args.count]
    print(f"Making {len(sentences)} clips in Virgo's voice...", flush=True)
    model = load_model(ref)
    settings, prefix = voice_settings(ref, "km")
    rate = model.tts_model.sample_rate
    kept = 0
    with open(manifest, "w", encoding="utf-8") as f:
        for i, text in enumerate(sentences):
            path = os.path.abspath(os.path.join(args.out, "clips", f"{i:05d}.wav"))
            if not os.path.exists(path):
                try:
                    wave = model.generate(text=prefix + text, cfg_value=2.0, inference_timesteps=10, **settings)
                except Exception as err:
                    print("skipped:", repr(err)[:120])
                    continue
                seconds = len(wave) / rate
                # Too long or too short for the words: the voice rambled or cut off. Leave it out.
                if not (1.0 <= seconds <= 15 and 0.04 <= seconds / len(text) <= 0.35):
                    continue
                sf.write(path, wave, rate)
            f.write(json.dumps({"audio": path, "text": text}, ensure_ascii=False) + "\n")
            kept += 1
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(sentences)} ({kept} kept)", flush=True)
    print(f"✅ {kept} clips")


if __name__ == "__main__":
    main()
