"""Virgo 1.0 speech: speech to text, long-audio transcription, and text to speech.

    python speech/virgo_speech.py stt recording.wav
    python speech/virgo_speech.py transcribe meeting.mp3 --srt meeting.srt
    python speech/virgo_speech.py tts "សួស្តី ខ្ញុំឈ្មោះ Virgo" out.wav

STT uses Whisper: Virgo's fine-tuned copy (speech/out/virgo-1.0-stt, from finetune_stt.py) when it
exists, otherwise openai/whisper-large-v3-turbo on a GPU (whisper-small on a CPU). TTS speaks Khmer with Virgo's own voice (khmer_voice.py, trained by train_voice.py) when it's set up, otherwise Meta MMS (facebook/mms-tts-khm), and
other text with Kokoro-82M (English, natural voices). Both run on a CPU.
"""
import argparse
import os
import re

import numpy as np
import soundfile as sf
import torch

KHMER = re.compile(r"[ក-៿᧠-᧿]")
STT_TUNED = "speech/out/virgo-1.0-stt"
DEVICE = 0 if torch.cuda.is_available() else -1
# Whisper large-v3-turbo hears Khmer far better than small; on a CPU, small stays (turbo is slow there).
STT_BASE = os.environ.get("VIRGO_STT_MODEL") or ("openai/whisper-large-v3-turbo" if DEVICE == 0 else "openai/whisper-small")
# Realtime voice picks only between these languages (Whisper codes), so Khmer isn't heard as another language.
STT_LANGUAGES = [c.strip() for c in os.environ.get("VIRGO_STT_LANGUAGES", "km,en").split(",") if c.strip()]
WHISPER_NAMES = {"km": "khmer", "en": "english"}


class VirgoSpeech:
    def __init__(self, stt_model=None):
        self.stt_model = stt_model or (STT_TUNED if os.path.isdir(STT_TUNED) else STT_BASE)
        self._stt = None
        self._khmer_tts = None
        self._english_tts = None

    # ---------- Speech to text ----------
    def _pipeline(self):
        if self._stt is None:
            from transformers import pipeline

            self._stt = pipeline("automatic-speech-recognition", model=self.stt_model, device=DEVICE)
        return self._stt

    def stt(self, audio, language=None):
        """Short audio (a file path, 16 kHz float samples, or {"raw", "sampling_rate"}) → text.
        `language` like "khmer"."""
        if isinstance(audio, (dict, np.ndarray)):
            return self._stt_samples(audio, language)
        kwargs = {"generate_kwargs": {"language": language, "task": "transcribe"}} if language else {}
        return self._pipeline()(audio, **kwargs)["text"].strip()

    @torch.inference_mode()
    def _stt_samples(self, audio, language=None):
        # Microphone samples go straight to Whisper: newer transformers pipelines fail on raw arrays.
        samples = audio["raw"] if isinstance(audio, dict) else audio
        rate = audio.get("sampling_rate", 16000) if isinstance(audio, dict) else 16000
        samples = np.asarray(samples, np.float32).reshape(-1)
        if rate != 16000:
            samples = np.interp(np.arange(0, len(samples), rate / 16000), np.arange(len(samples)), samples).astype(np.float32)
        pipe = self._pipeline()
        model = pipe.model
        features = pipe.feature_extractor(samples, sampling_rate=16000, return_tensors="pt").input_features
        features = features.to(model.device, next(model.parameters()).dtype)
        language = language or self._pick_language(model, pipe.tokenizer, features)
        kwargs = {"language": language, "task": "transcribe"} if language else {}
        ids = model.generate(features, **kwargs)
        return pipe.tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()

    @staticmethod
    def _pick_language(model, tokenizer, features):
        """The most likely of STT_LANGUAGES (Whisper's own language guess, limited to those)."""
        codes = [c for c in STT_LANGUAGES if tokenizer.convert_tokens_to_ids(f"<|{c}|>") not in (None, tokenizer.unk_token_id)]
        if len(codes) < 2:
            return WHISPER_NAMES.get(codes[0], codes[0]) if codes else None
        try:
            start = torch.tensor([[model.generation_config.decoder_start_token_id]], device=model.device)
            logits = model(input_features=features, decoder_input_ids=start).logits[0, -1]
            ids = [tokenizer.convert_tokens_to_ids(f"<|{c}|>") for c in codes]
            best = codes[int(torch.argmax(logits[ids]))]
            return WHISPER_NAMES.get(best, best)
        except Exception as err:  # fall back to Whisper's free choice
            print("Virgo speech: language pick failed:", repr(err))
            return None

    # ---------- Transcribe (long audio, with timestamps) ----------
    def transcribe(self, path, language=None):
        """Long audio → [{start, end, text}] in 30-second chunks."""
        kwargs = {"generate_kwargs": {"language": language, "task": "transcribe"}} if language else {}
        out = self._pipeline()(path, chunk_length_s=30, batch_size=8, return_timestamps=True, **kwargs)
        return [{"start": c["timestamp"][0] or 0.0, "end": c["timestamp"][1] or c["timestamp"][0] or 0.0, "text": c["text"].strip()}
                for c in out["chunks"]]

    @staticmethod
    def to_srt(segments):
        def ts(sec):
            ms = int(round(sec * 1000))
            return f"{ms // 3_600_000:02}:{ms // 60_000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"
        return "\n".join(f"{i}\n{ts(s['start'])} --> {ts(s['end'])}\n{s['text']}\n" for i, s in enumerate(segments, 1))

    # ---------- Text to speech ----------
    def tts(self, text, voice="af_heart"):
        """Text → (samples float32, sample_rate). Virgo-1.0-Angkor-Voice (VoxCPM2, vox_voice.py) speaks
        every language when it's set up; otherwise Khmer text uses the Khmer voice and English Kokoro."""
        vox = self._vox_voice()
        if vox:
            try:
                return vox(text)
            except Exception as err:
                print("Virgo-1.0-Angkor-Voice failed, using the other voices for now:", err)
                self._vox = False
        if KHMER.search(text):
            return self._khmer(text)
        return self._english(text, voice)

    def _vox_voice(self):
        if getattr(self, "_vox", None) is None:
            self._vox = False
            try:
                import vox_voice

                if vox_voice.available():
                    self._vox = vox_voice.VoxVoice()
            except Exception as err:
                print("Virgo-1.0-Angkor-Voice isn't available:", err)
        return self._vox

    @staticmethod
    def _mms(name):
        from transformers import AutoTokenizer, VitsModel

        return AutoTokenizer.from_pretrained(name), VitsModel.from_pretrained(name).eval()

    @staticmethod
    def _speak_mms(pair, text):
        tok, model = pair
        with torch.inference_mode():
            wave = model(**tok(text, return_tensors="pt")).waveform[0].numpy()
        return wave.astype(np.float32), model.config.sampling_rate

    def _khmer(self, text):
        # Virgo's own Khmer voice (speech/train_voice.py, commercial-safe) when it's set up; otherwise
        # Meta MMS, which is non-commercial (CC BY-NC 4.0).
        if self._khmer_tts is None:
            self._khmer_tts = self._own_khmer_voice() or ("mms", self._mms("facebook/mms-tts-khm"))
        kind, engine = self._khmer_tts
        if kind == "virgo":
            try:
                return engine(text)
            except Exception as err:
                print("Virgo's Khmer voice failed, using MMS for now:", err)
                self._khmer_tts = ("mms", self._mms("facebook/mms-tts-khm"))
                return self._speak_mms(self._khmer_tts[1], text)
        return self._speak_mms(engine, text)

    @staticmethod
    def _own_khmer_voice():
        try:
            import khmer_voice

            if khmer_voice.available():
                return ("virgo", khmer_voice.KhmerVoice())
        except Exception as err:
            print("Virgo's own Khmer voice isn't available, using MMS:", err)
        return None

    def _english(self, text, voice):
        # Kokoro sounds most natural; where it can't be installed (e.g. Python 3.13 on Colab), Meta's
        # MMS English voice is used instead.
        if self._english_tts is None:
            try:
                from kokoro import KPipeline

                self._english_tts = ("kokoro", KPipeline(lang_code="a"))  # American English
            except Exception:
                self._english_tts = ("mms", self._mms("facebook/mms-tts-eng"))
        kind, engine = self._english_tts
        if kind == "mms":
            return self._speak_mms(engine, text)
        parts = [audio for _, _, audio in engine(text, voice=voice)]
        return np.concatenate(parts).astype(np.float32), 24000


def main():
    p = argparse.ArgumentParser(description="Virgo 1.0 speech")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stt"); s.add_argument("audio"); s.add_argument("--language")
    t = sub.add_parser("transcribe"); t.add_argument("audio"); t.add_argument("--language"); t.add_argument("--srt")
    v = sub.add_parser("tts"); v.add_argument("text"); v.add_argument("out"); v.add_argument("--voice", default="af_heart")
    args = p.parse_args()
    speech = VirgoSpeech()
    if args.cmd == "stt":
        print(speech.stt(args.audio, args.language))
    elif args.cmd == "transcribe":
        segments = speech.transcribe(args.audio, args.language)
        for seg in segments:
            print(f"[{seg['start']:7.1f}s] {seg['text']}")
        if args.srt:
            with open(args.srt, "w", encoding="utf-8") as f:
                f.write(speech.to_srt(segments))
    else:
        wave, rate = speech.tts(args.text, args.voice)
        sf.write(args.out, wave, rate)
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
