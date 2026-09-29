"""Train Virgo's own Khmer voice (text to speech) from scratch: a VITS model, safe for commercial use.

    python speech/train_voice.py --push <your-hf-name>/Virgo-1.0-Angkor-Voice
    python speech/train_voice.py --extra my_recordings/ --hours 11

Default data: OpenSLR 42, Google's crowdsourced high-quality Khmer speech (CC BY-SA 4.0, commercial
use allowed with attribution). Add your own recordings with --extra: a folder of .wav files and a
metadata.csv of `file_name,sentence` (the same format as finetune_stt.py). No base voice is used, so
there's no non-commercial license (unlike Meta MMS).

A good voice needs many hours of training, more than one Kaggle session (12 h). The script stops
itself before --hours, saves, and uploads the run to Hugging Face (--push); run it again and it
continues where it stopped. The trained voice goes to speech/out/virgo-khmer-voice, where
virgo_speech.py uses it instead of MMS.
"""
import argparse
import csv
import glob
import json
import os
import shutil
import time
import unicodedata
import urllib.request
import zipfile

import numpy as np
import soundfile as sf

RATE = 22050
SLR42 = [f"https://{host}/resources/42/km_kh_male.zip" for host in ("www.openslr.org", "us.openslr.org", "openslr.elda.org")]
PUNCTUATION = "!'(),-.:;? ។៕៖«»\"…"
TEST_SENTENCES = ["សួស្តី ខ្ញុំឈ្មោះ Virgo ជាជំនួយការ AI របស់ KSN។", "តើអ្នកសុខសប្បាយជាទេ?", "អរគុណច្រើន។"]


def clean(text):
    return " ".join(unicodedata.normalize("NFC", text).replace("​", "").split())


def download_slr42(root):
    """OpenSLR 42 → [(speaker, wav path, text)]."""
    folder = os.path.join(root, "slr42")
    if not glob.glob(os.path.join(folder, "**", "line_index.tsv"), recursive=True):
        os.makedirs(folder, exist_ok=True)
        zip_path = os.path.join(root, "slr42.zip")
        for url in SLR42:
            try:
                print("⬇️ Downloading", url)
                urllib.request.urlretrieve(url, zip_path)
                break
            except Exception as err:
                print("   failed:", err)
        else:
            raise SystemExit("❌ Couldn't download OpenSLR 42 (is internet on?).")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(folder)
        os.remove(zip_path)
    wavs = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(folder, "**", "*.wav"), recursive=True)}
    rows = []
    for index in glob.glob(os.path.join(folder, "**", "line_index.tsv"), recursive=True):
        with open(index, encoding="utf-8") as f:
            for line in f:
                parts = [p.strip() for p in line.rstrip("\n").split("\t") if p.strip()]
                if len(parts) >= 2 and parts[0] in wavs:
                    # File names look like khm_0308_0011865648: the second part is the speaker.
                    speaker = parts[0].split("_")[1] if parts[0].count("_") >= 2 else "slr42"
                    rows.append((f"slr42_{speaker}", wavs[parts[0]], clean(parts[-1])))
    return rows


def read_extra(folder):
    """Your own recordings: metadata.csv with file_name,sentence → [(speaker, wav path, text)]."""
    rows = []
    with open(os.path.join(folder, "metadata.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(("mine", os.path.join(folder, row["file_name"]), clean(row["sentence"])))
    return rows


def prepare(rows, out):
    """Mono 22.05 kHz wavs with silence trimmed, and metadata.csv (file|text|speaker)."""
    import librosa

    wav_dir = os.path.join(out, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    meta = os.path.join(out, "metadata.csv")
    if os.path.exists(meta):
        return meta
    kept = []
    for i, (speaker, path, text) in enumerate(rows):
        if not text:
            continue
        try:
            wave, _ = librosa.load(path, sr=RATE, mono=True)
        except Exception as err:
            print("skip", path, err)
            continue
        wave, _ = librosa.effects.trim(wave, top_db=35)
        seconds = len(wave) / RATE
        if not 0.5 <= seconds <= 15:
            continue
        name = f"{i:06d}.wav"
        sf.write(os.path.join(wav_dir, name), (wave / max(1e-6, np.abs(wave).max()) * 0.95).astype(np.float32), RATE)
        kept.append((name, text, speaker))
    with open(meta, "w", encoding="utf-8") as f:
        for name, text, speaker in kept:
            f.write(f"{name}|{text}|{speaker}\n")
    print(f"✅ {len(kept)} clips ready ({len({s for _, _, s in kept})} speakers)")
    return meta


def formatter(root_path, meta_file, **_):
    with open(os.path.join(root_path, meta_file), encoding="utf-8") as f:
        return [{"text": t, "audio_file": os.path.join(root_path, "wavs", n), "speaker_name": s, "root_path": root_path, "language": "km"}
                for n, t, s in (line.rstrip("\n").split("|") for line in f if line.strip())]


def pull(repo, run_dir):
    """The last uploaded run (config.json + checkpoint) from Hugging Face, to continue it."""
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(repo, local_dir=run_dir, allow_patterns=["config.json", "checkpoint_*.pth", "speakers.pth"])
    except Exception as err:
        print("Starting a new voice (nothing to continue on Hugging Face yet):", type(err).__name__)
    return bool(glob.glob(os.path.join(run_dir, "checkpoint_*.pth")))


def push(repo, run_dir, final=False, out=None):
    """Uploads the newest checkpoint (and, at the end, the voice itself: model.pth + config.json)."""
    from huggingface_hub import HfApi

    checkpoints = sorted(glob.glob(os.path.join(run_dir, "checkpoint_*.pth")), key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
    if not checkpoints:
        return
    api = HfApi()
    api.create_repo(repo, private=True, exist_ok=True)
    files = {"config.json": os.path.join(run_dir, "config.json"), os.path.basename(checkpoints[-1]): checkpoints[-1]}
    if os.path.exists(os.path.join(run_dir, "speakers.pth")):
        files["speakers.pth"] = os.path.join(run_dir, "speakers.pth")
    if final and out:
        for name in ("model.pth", "voice.json"):
            files[name] = os.path.join(out, name)
    for name, path in files.items():
        api.upload_file(path_or_fileobj=path, path_in_repo=name, repo_id=repo)
    # Only the newest checkpoint stays on Hugging Face.
    for old in api.list_repo_files(repo):
        if old.startswith("checkpoint_") and old != os.path.basename(checkpoints[-1]):
            api.delete_file(old, repo_id=repo)
    print("☁️ Uploaded", os.path.basename(checkpoints[-1]), "to", repo)


def main():
    p = argparse.ArgumentParser(description="Train Virgo's own Khmer voice")
    p.add_argument("--work", default="speech/work/voice", help="where the prepared clips and the run go")
    p.add_argument("--out", default="speech/out/virgo-khmer-voice")
    p.add_argument("--extra", help="your own recordings: .wav files + metadata.csv (file_name,sentence)")
    p.add_argument("--no-slr42", action="store_true", help="train only on --extra")
    p.add_argument("--hours", type=float, default=11.0, help="stop, save and upload after this long")
    p.add_argument("--push", help="Hugging Face repo to save to and continue from, e.g. you/Virgo-1.0-Angkor-Voice")
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--save-step", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=100000)
    args = p.parse_args()

    from trainer import Trainer, TrainerArgs
    from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
    from TTS.tts.configs.vits_config import VitsConfig
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig
    from TTS.tts.utils.speakers import SpeakerManager
    from TTS.tts.utils.text.tokenizer import TTSTokenizer
    from TTS.utils.audio import AudioProcessor

    started = time.time()
    data_dir = os.path.join(args.work, "data")
    run_dir = os.path.join(args.work, "run")
    rows = [] if args.no_slr42 else download_slr42(args.work)
    if args.extra:
        rows += read_extra(args.extra)
    meta = prepare(rows, data_dir)

    samples = formatter(data_dir, "metadata.csv")
    from TTS.tts.utils.text.cleaners import multilingual_cleaners

    # Every character the (cleaned, lower-cased) text uses, plus the test sentences'.
    texts = [s["text"] for s in samples] + TEST_SENTENCES
    letters = sorted({c for t in texts for c in multilingual_cleaners(t)} - set(PUNCTUATION))
    dataset = BaseDatasetConfig(formatter="virgo", meta_file_train=os.path.basename(meta), path=data_dir, language="km")
    config = VitsConfig(
        run_name="virgo-khmer-voice",
        audio=VitsAudioConfig(sample_rate=RATE, win_length=1024, hop_length=256, num_mels=80, mel_fmin=0, mel_fmax=None),
        model_args=VitsArgs(use_speaker_embedding=True),
        batch_size=args.batch,
        eval_batch_size=8,
        batch_group_size=5,
        num_loader_workers=2,
        num_eval_loader_workers=1,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=args.epochs,
        text_cleaner="multilingual_cleaners",
        use_phonemes=False,
        characters=CharactersConfig(
            characters_class="TTS.tts.models.vits.VitsCharacters",
            characters="".join(letters),
            punctuations=PUNCTUATION,
            pad="<PAD>",
            eos="<EOS>",
            bos="<BOS>",
            blank="<BLNK>",
        ),
        compute_input_seq_cache=True,
        print_step=50,
        print_eval=False,
        save_step=args.save_step,
        save_n_checkpoints=2,
        save_checkpoints=True,
        save_best_after=10**9,  # keep only step checkpoints: they're what continuing needs
        mixed_precision=False,
        output_path=run_dir,
        datasets=[dataset],
        test_sentences=[[s, None, None, "km"] for s in TEST_SENTENCES],
        cudnn_benchmark=False,
    )

    # Continue the last run (from Hugging Face, or this folder) when there is one.
    previous = os.path.join(run_dir, "continue")
    if args.push:
        pull(args.push, previous)
    can_continue = bool(glob.glob(os.path.join(previous, "checkpoint_*.pth")))
    if can_continue:
        print("▶️ Continuing the last training run")
        config.load_json(os.path.join(previous, "config.json"))
        config.output_path = run_dir

    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)
    train, evals = load_tts_samples(dataset, eval_split=True, formatter=formatter, eval_split_max_size=64, eval_split_size=min(0.2, max(0.02, 2 / len(samples))))
    speakers = SpeakerManager()
    speakers.set_ids_from_data(train + evals, parse_key="speaker_name")
    config.model_args.num_speakers = speakers.num_speakers
    config.num_speakers = speakers.num_speakers
    model = Vits(config, ap, tokenizer, speakers)

    deadline = started + args.hours * 3600

    class TimeUp(Exception):
        pass

    def on_step(trainer):
        if time.time() > deadline:
            print(f"⏱️ {args.hours} hours reached: saving and stopping.")
            trainer.save_checkpoint()
            raise TimeUp

    trainer = Trainer(
        TrainerArgs(continue_path=previous if can_continue else ""),
        config,
        run_dir,
        model=model,
        train_samples=train,
        eval_samples=evals,
        callbacks={"on_train_step_end": on_step},
    )
    # trainer.fit() would end the whole process on a stop; its loop is run directly instead.
    try:
        trainer._fit()
    except TimeUp:
        pass
    finally:
        trainer.dashboard_logger.finish()

    # The trainer's own folder (a new one, or the continued one) holds the latest checkpoint.
    latest = trainer.output_path
    checkpoints = sorted(glob.glob(os.path.join(latest, "checkpoint_*.pth")), key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
    if not checkpoints:
        raise SystemExit("❌ No checkpoint was saved.")
    os.makedirs(args.out, exist_ok=True)
    newest = checkpoints[-1]
    shutil.copy(newest, os.path.join(args.out, "model.pth"))
    shutil.copy(os.path.join(latest, "config.json"), os.path.join(args.out, "config.json"))
    if os.path.exists(os.path.join(latest, "speakers.pth")):
        shutil.copy(os.path.join(latest, "speakers.pth"), os.path.join(args.out, "speakers.pth"))
    # The voice speaks as the speaker with the most clips (the clearest, best-learned voice).
    counts = {}
    for sample in samples:
        counts[sample["speaker_name"]] = counts.get(sample["speaker_name"], 0) + 1
    with open(os.path.join(args.out, "voice.json"), "w", encoding="utf-8") as f:
        json.dump({"speaker": max(counts, key=counts.get), "step": int(os.path.basename(newest).split("_")[1].split(".")[0])}, f)
    print("✅ Virgo's Khmer voice saved to", args.out, f"({os.path.basename(newest)})")
    if args.push:
        push(args.push, latest, final=True, out=args.out)


if __name__ == "__main__":
    main()
