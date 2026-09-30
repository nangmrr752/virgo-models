"""Virgo's own Khmer voice (trained by train_voice.py): text → (samples, sample rate).

The voice runs in its own Python environment (coqui-tts needs older transformers/torch than the Virgo
server), as a small worker process the server starts once:

    VIRGO_VOICE_PYTHON=/path/to/voice-env/bin/python   # that environment has coqui-tts installed
    VIRGO_VOICE_REPO=you/Virgo-1.0-Angkor-Voice        # or put the voice in speech/out/virgo-khmer-voice

The worker (`python speech/khmer_voice.py --serve <folder>`) reads one JSON line per sentence on stdin
and answers with one JSON line naming a WAV file it wrote.
"""
import json
import os
import subprocess
import sys
import tempfile
import threading

VOICE_DIR = os.environ.get("VIRGO_VOICE_DIR", "speech/out/virgo-khmer-voice")
FILES = ["model.pth", "config.json", "speakers.pth", "voice.json"]


def available():
    """True when a voice environment is set up and a trained voice exists (here or on Hugging Face)."""
    return bool(os.environ.get("VIRGO_VOICE_PYTHON")) and (
        os.path.exists(os.path.join(VOICE_DIR, "model.pth")) or bool(os.environ.get("VIRGO_VOICE_REPO")))


class KhmerVoice:
    def __init__(self, folder=VOICE_DIR):
        folder = os.path.abspath(folder)
        if not os.path.exists(os.path.join(folder, "model.pth")) and os.environ.get("VIRGO_VOICE_REPO"):
            from huggingface_hub import snapshot_download

            snapshot_download(os.environ["VIRGO_VOICE_REPO"], local_dir=folder, allow_patterns=FILES)
        python = os.environ.get("VIRGO_VOICE_PYTHON", sys.executable)
        self.lock = threading.Lock()
        self.worker = subprocess.Popen([python, os.path.abspath(__file__), "--serve", folder],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8")
        ready = json.loads(self.worker.stdout.readline() or '{"error": "the voice worker stopped"}')
        if "error" in ready:
            raise RuntimeError(f"Virgo Khmer voice: {ready['error']}")
        print("Virgo's own Khmer voice is ready:", ready)

    def __call__(self, text):
        import soundfile as sf

        with self.lock:
            if self.worker.poll() is not None:
                raise RuntimeError("the Khmer voice worker stopped")
            self.worker.stdin.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            self.worker.stdin.flush()
            answer = json.loads(self.worker.stdout.readline() or '{"error": "no answer"}')
        if "error" in answer:
            raise RuntimeError(f"Virgo Khmer voice: {answer['error']}")
        try:
            wave, rate = sf.read(answer["path"], dtype="float32")
        finally:
            os.remove(answer["path"])
        return wave, rate


def serve(folder):
    """The worker: loads the voice once, then speaks each line of text it's sent."""
    protocol = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)  # everything the libraries print goes to the log, not the protocol

    def say(**data):
        protocol.write(json.dumps(data, ensure_ascii=False) + "\n")

    try:
        import soundfile as sf
        import torch
        from TTS.tts.configs.vits_config import VitsConfig
        from TTS.tts.models.vits import Vits

        config = VitsConfig()
        config.load_json(os.path.join(folder, "config.json"))
        # The saved config points at the training folder's speakers file: use this folder's.
        speakers = os.path.join(folder, "speakers.pth")
        config.speakers_file = speakers
        config.model_args.speakers_file = speakers
        model = Vits.init_from_config(config)
        model.load_checkpoint(config, os.path.join(folder, "model.pth"), eval=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        names = model.speaker_manager.speaker_names if model.speaker_manager else []
        try:
            with open(os.path.join(folder, "voice.json"), encoding="utf-8") as f:
                wanted = json.load(f).get("speaker")
        except FileNotFoundError:
            wanted = None
        speaker = wanted if wanted in names else (names[0] if names else None)
        rate = config.audio.sample_rate
    except Exception as err:
        say(error=repr(err))
        return
    say(ready=True, speaker=speaker, device=device, rate=rate)

    for line in sys.stdin:
        try:
            text = json.loads(line)["text"]
            ids = torch.LongTensor(model.tokenizer.text_to_ids(text)).unsqueeze(0).to(device)
            aux = {"speaker_ids": torch.LongTensor([model.speaker_manager.name_to_id[speaker]]).to(device)} if speaker else {}
            with torch.inference_mode():
                wave = model.inference(ids, aux_input=aux)["model_outputs"].squeeze().float().cpu().numpy()
            path = tempfile.mktemp(suffix=".wav")
            sf.write(path, wave, rate, subtype="PCM_16")
            say(path=path, rate=rate)
        except Exception as err:
            say(error=repr(err))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--serve":
        serve(sys.argv[2])
    else:
        sys.exit("usage: python speech/khmer_voice.py --serve <voice folder>")
