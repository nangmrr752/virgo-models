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


def clean(wave, rate):
    """Cleaner audio: silence trimmed at both ends, loudness evened out (peak at -1 dBFS)."""
    import numpy as np

    wave = np.asarray(wave, np.float32)
    loud = np.flatnonzero(np.abs(wave) > 0.01)
    if len(loud):
        pad = int(0.12 * rate)
        wave = wave[max(0, loud[0] - pad): loud[-1] + pad]
    peak = float(np.max(np.abs(wave))) if len(wave) else 0
    return wave * (0.89 / peak) if peak > 0 else wave


class Checker:
    """Listens to each clip with Whisper (Virgo's hearing) and scores how many Khmer letters it got wrong
    (character error rate): a clip the voice mispronounced, mumbled or garbled scores badly."""

    def __init__(self, model):
        import torch
        from transformers import pipeline

        print("Checking clips with", model, flush=True)
        self.asr = pipeline("automatic-speech-recognition", model=model, torch_dtype=torch.float16, device=0 if torch.cuda.is_available() else -1)

    @staticmethod
    def _norm(text):
        return re.sub(r"[\s\u200b។៕,.!?;:'\"()«»…-]", "", text)

    def cer(self, wave, rate, text):
        import numpy as np

        if rate != 16000:
            wave = np.interp(np.arange(0, len(wave), rate / 16000), np.arange(len(wave)), wave).astype(np.float32)
        heard = self.asr({"raw": wave, "sampling_rate": 16000}, generate_kwargs={"language": "khmer", "task": "transcribe"})["text"]
        a, b = self._norm(text), self._norm(heard)
        if not a:
            return 1.0
        prev = list(range(len(b) + 1))  # edit distance, row by row
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1] / len(a)


def design_voice(ref, repo, description, tries=4, checker=None, steps=10):
    """A new voice from a description: VoxCPM2 speaks one Khmer sentence in it, and that clip becomes the
    voice sample (voice.wav + voice.json), saved to `repo`, so every later sentence sounds the same."""
    import soundfile as sf
    from huggingface_hub import HfApi

    print(f"Making a new voice: {description}", flush=True)
    model = load_model(ref)
    best = None
    rate = model.tts_model.sample_rate
    for seed in range(tries):  # several tries; keep the clearest (fewest wrong letters), then the most natural length
        import torch

        torch.manual_seed(seed)
        wave = clean(model.generate(text=f"({description}){DESIGN_TEXT}", cfg_value=2.0, inference_timesteps=steps), rate)
        seconds = len(wave) / rate
        if not 3 <= seconds <= 12:
            continue
        score = (checker.cer(wave, rate, DESIGN_TEXT) if checker else 0) + abs(seconds - 6) / 100
        print(f"  try {seed + 1}: {seconds:.1f} s, score {score:.3f}", flush=True)
        if best is None or score < best[1]:
            best = (wave, score)
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
    p.add_argument("--steps", type=int, default=10, help="VoxCPM2 steps per clip: more = cleaner, slower")
    p.add_argument("--check", help="listen to each clip with this Whisper model and drop mispronounced ones")
    p.add_argument("--max-cer", type=float, default=0.15, help="with --check: drop clips with more wrong letters than this")
    p.add_argument("--design-tries", type=int, default=4)
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

    checker = Checker(args.check) if args.check else None
    ref = os.path.join(args.out, "reference")  # only the voice sample: no old LoRA while making clips
    try:
        snapshot_download(args.repo, local_dir=ref, allow_patterns=["voice.wav", "voice.json"])
    except Exception:  # a new voice: its repo doesn't exist yet
        os.makedirs(ref, exist_ok=True)
    if not os.path.exists(os.path.join(ref, "voice.wav")):
        if not args.design:
            sys.exit("❌ No voice.wav in " + args.repo + ": make Virgo's voice first (speech/design_voice.ipynb).")
        design_voice(ref, args.repo, args.design, args.design_tries, checker, args.steps)
    sentences = khmer_sentences(args.data)
    random.Random(7).shuffle(sentences)
    sentences = sentences[: args.count]
    print(f"Making {len(sentences)} clips in Virgo's voice...", flush=True)
    model = load_model(ref)
    settings, prefix = voice_settings(ref, "km")
    rate = model.tts_model.sample_rate
    kept = dropped = 0
    with open(manifest, "w", encoding="utf-8") as f:
        for i, text in enumerate(sentences):
            path = os.path.abspath(os.path.join(args.out, "clips", f"{i:05d}.wav"))
            if not os.path.exists(path):
                try:
                    wave = clean(model.generate(text=prefix + text, cfg_value=2.0, inference_timesteps=args.steps, **settings), rate)
                except Exception as err:
                    print("skipped:", repr(err)[:120])
                    continue
                seconds = len(wave) / rate
                # Too long or too short for the words: the voice rambled or cut off. Leave it out.
                if not (1.0 <= seconds <= 15 and 0.04 <= seconds / len(text) <= 0.35):
                    continue
                # Said wrong: Whisper hears different words than the text. Leave it out.
                if checker and checker.cer(wave, rate, text) > args.max_cer:
                    dropped += 1
                    continue
                sf.write(path, wave, rate)
            f.write(json.dumps({"audio": path, "text": text}, ensure_ascii=False) + "\n")
            kept += 1
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(sentences)} ({kept} kept{f', {dropped} mispronounced dropped' if checker else ''})", flush=True)
    print(f"✅ {kept} clips" + (f" ({dropped} mispronounced ones left out)" if checker else ""))


if __name__ == "__main__":
    main()
