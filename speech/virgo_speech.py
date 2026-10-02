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
# Names Whisper should expect and spell right (VIRGO_STT_WORDS to change; empty to turn off).
STT_WORDS = "Virgo AI, KSN, Virgo-1.0-Bayon, Angkor, កម្ពុជា, ភ្នំពេញ."
# Whisper large-v3-turbo hears Khmer far better than small; on a CPU, small stays (turbo is slow there).
STT_BASE = os.environ.get("VIRGO_STT_MODEL") or ("openai/whisper-large-v3-turbo" if DEVICE == 0 else "openai/whisper-small")
# Realtime voice picks only between these languages (Whisper codes), so Khmer isn't heard as another language.
# Which languages hearing listens for: "all" (every language Whisper knows, ~99), or a list like "km,en".
STT_LANGUAGES = [c.strip() for c in os.environ.get("VIRGO_STT_LANGUAGES", "all").split(",") if c.strip()]
WHISPER_NAMES = {"km": "khmer", "en": "english"}
# Virgo's own hearing (STT_TUNED) was trained only on Khmer and now hears everything as Khmer. So a
# second, general Whisper (every language, ~99) decides which language is spoken and writes down all
# languages but Khmer; Virgo's own hearing writes down Khmer. VIRGO_STT_GENERAL_MODEL=off: one model only.
STT_GENERAL = os.environ.get("VIRGO_STT_GENERAL_MODEL") or STT_BASE
KHMER_NAMES = ("km", "khmer")


# Scripts the fallback voices can tell apart, in text order. Latin letters are "en" (Microsoft's
# multilingual English voices also read French, Spanish, Vietnamese... well).
SCRIPTS = [
    ("km", r"\u1780-\u17ff\u19e0-\u19ff"), ("th", r"\u0e00-\u0e7f"), ("lo", r"\u0e80-\u0eff"), ("my", r"\u1000-\u109f"),
    ("ja", r"\u3040-\u30ff"), ("ko", r"\uac00-\ud7af\u1100-\u11ff"), ("zh", r"\u4e00-\u9fff\u3400-\u4dbf"),
    ("hi", r"\u0900-\u097f"), ("ar", r"\u0600-\u06ff"), ("ru", r"\u0400-\u04ff"), ("en", r"A-Za-z\u00c0-\u024f\u1e00-\u1eff"),
]
SCRIPT_OF = [(lang, re.compile(f"[{chars}]")) for lang, chars in SCRIPTS]


def char_language(c):
    return next((lang for lang, pattern in SCRIPT_OF if pattern.match(c)), None)


def split_languages(text):
    """Text → [(lang, part)] in order, one part per writing system (Khmer, Thai, Chinese, Latin...).
    Numbers, punctuation and single Latin letters stay with the part around them."""
    parts = []
    for c in text:
        lang = char_language(c)
        if lang is None or not parts:
            if not parts:
                parts.append([lang, c])
            else:
                parts[-1][1] += c
            continue
        if parts[-1][0] is None:
            parts[-1][0] = lang
        if parts[-1][0] == lang:
            parts[-1][1] += c
        else:
            parts.append([lang, c])
    # A Latin bit of one letter (an "A", a "x") inside another language stays with it.
    merged = []
    for lang, part in parts:
        if merged and (lang == merged[-1][0] or (lang == "en" and sum(ch.isalpha() for ch in part) < 2 and merged[-1][0] != "en")):
            merged[-1][1] += part
        else:
            merged.append([lang, part])
    # A single letter at the very start goes with what follows ("A ជាអក្សរ" is Khmer).
    if len(merged) > 1 and merged[0][0] == "en" and sum(ch.isalpha() for ch in merged[0][1]) < 2:
        merged[1][1] = merged[0][1] + merged[1][1]
        merged.pop(0)
    # Chinese characters in Japanese text (kanji) are Japanese.
    if any(lang == "ja" for lang, _ in merged):
        merged = [["ja" if lang == "zh" else lang, part] for lang, part in merged]
    out = [((lang or "en"), part.strip()) for lang, part in merged if re.search(r"\w", part)]
    joined = []
    for lang, part in out:  # neighbours that became the same language
        if joined and joined[-1][0] == lang:
            joined[-1] = (lang, joined[-1][1] + " " + part)
        else:
            joined.append((lang, part))
    out = joined
    return out or [("km" if KHMER.search(text) else "en", text)]


class VirgoSpeech:
    def __init__(self, stt_model=None):
        self.stt_model = stt_model or (STT_TUNED if os.path.isdir(STT_TUNED) else STT_BASE)
        self.general_model = None if STT_GENERAL.lower() == "off" else STT_GENERAL
        if self.general_model == self.stt_model:
            self.general_model = None  # no Khmer-only model here: the one model does everything
        self._stt = None
        self._general = None
        self._khmer_tts = None
        self._english_tts = None

    # ---------- Speech to text ----------
    def _pipeline(self, general=False):
        """Virgo's own (Khmer) hearing, or with general=True the every-language Whisper (the same model
        when there's no separate Khmer one)."""
        from transformers import pipeline

        dtype = torch.float16 if DEVICE == 0 else torch.float32
        if general and self.general_model:
            if self._general is None:
                print("Hearing: every language with", self.general_model, "+ Khmer with", self.stt_model)
                self._general = pipeline("automatic-speech-recognition", model=self.general_model, device=DEVICE, torch_dtype=dtype)
            return self._general
        if self._stt is None:
            self._stt = pipeline("automatic-speech-recognition", model=self.stt_model, device=DEVICE, torch_dtype=dtype)
        return self._stt

    def stt(self, audio, language=None):
        """Short audio (a file path, 16 kHz float samples, or {"raw", "sampling_rate"}) → text.
        `language` like "khmer"."""
        if isinstance(audio, (dict, np.ndarray)):
            return self._stt_samples(audio, language)
        if language:
            khmer = language in KHMER_NAMES
            return self._pipeline(general=not khmer)(audio, generate_kwargs={"language": language, "task": "transcribe"})["text"].strip()
        # A file in an unknown language: the general Whisper writes it down; if that's Khmer, Virgo's own
        # Khmer hearing writes it again, better.
        text = self._pipeline(general=True)(audio)["text"].strip()
        if self.general_model and KHMER.search(text):
            text = self._pipeline()(audio, generate_kwargs={"language": "khmer", "task": "transcribe"})["text"].strip()
        return text

    @torch.inference_mode()
    def _stt_samples(self, audio, language=None):
        # Microphone samples go straight to Whisper: newer transformers pipelines fail on raw arrays.
        samples = audio["raw"] if isinstance(audio, dict) else audio
        rate = audio.get("sampling_rate", 16000) if isinstance(audio, dict) else 16000
        samples = np.asarray(samples, np.float32).reshape(-1)
        if rate != 16000:
            samples = np.interp(np.arange(0, len(samples), rate / 16000), np.arange(len(samples)), samples).astype(np.float32)
        def features_for(pipe):
            f = pipe.feature_extractor(samples, sampling_rate=16000, return_tensors="pt").input_features
            return f.to(pipe.model.device, next(pipe.model.parameters()).dtype)

        if not language:  # the general Whisper decides the language (Virgo's Khmer hearing would say Khmer)
            judge = self._pipeline(general=True)
            language = self._pick_language(judge.model, judge.tokenizer, features_for(judge))
        pipe = self._pipeline(general=language not in KHMER_NAMES)
        model = pipe.model
        features = features_for(pipe)
        kwargs = {"language": language, "task": "transcribe"} if language else {}
        # Careful listening: Whisper weighs several guesses (beam search) instead of taking the first.
        beams = int(os.environ.get("VIRGO_STT_BEAMS", "5"))
        if beams > 1:
            kwargs["num_beams"] = beams
        # Virgo's own words (names it should spell right), given to Whisper as earlier "speech".
        words = os.environ.get("VIRGO_STT_WORDS", STT_WORDS).strip()
        if language not in (None, "khmer", "english", "km", "en"):
            words = ""  # Virgo's word list is Khmer and English: it would only confuse other languages
        if words:
            try:
                kwargs["prompt_ids"] = pipe.tokenizer.get_prompt_ids(words, return_tensors="pt").to(model.device)
            except Exception:
                pass
        ids = model.generate(features, **kwargs)
        text = pipe.tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()
        if words and text.startswith(words):  # older transformers keep the prompt in the output
            text = text[len(words):].strip()
        return text

    @staticmethod
    def _pick_language(model, tokenizer, features):
        """The most likely of STT_LANGUAGES (Whisper's own language guess, limited to those)."""
        wanted = STT_LANGUAGES
        if wanted == ["all"]:
            lang_to_id = getattr(model.generation_config, "lang_to_id", None) or {}
            wanted = [t.strip("<|>") for t in lang_to_id] or ["km", "en"]
        codes = [c for c in wanted if tokenizer.convert_tokens_to_ids(f"<|{c}|>") not in (None, tokenizer.unk_token_id)]
        if len(codes) < 2:
            return WHISPER_NAMES.get(codes[0], codes[0]) if codes else None
        try:
            start = torch.tensor([[model.generation_config.decoder_start_token_id]], device=model.device)
            logits = model(input_features=features, decoder_input_ids=start).logits[0, -1]
            ids = [tokenizer.convert_tokens_to_ids(f"<|{c}|>") for c in codes]
            scores = logits[ids]
            best = codes[int(torch.argmax(scores))]
            # Khmer mixed with English words is still Khmer: unless another language is clearly more
            # likely (VIRGO_STT_KHMER_BIAS, default 2.0 logits), Khmer wins.
            if "km" in codes and best != "km":
                bias = float(os.environ.get("VIRGO_STT_KHMER_BIAS", "2.0"))
                if float(scores.max() - scores[codes.index("km")]) < bias:
                    best = "km"
            return WHISPER_NAMES.get(best, best)
        except Exception as err:  # fall back to Whisper's free choice
            print("Virgo speech: language pick failed:", repr(err))
            return None

    # ---------- Transcribe (long audio, with timestamps) ----------
    def transcribe(self, path, language=None):
        """Long audio → [{start, end, text}] in 30-second chunks."""
        def run(pipe, lang):
            kwargs = {"generate_kwargs": {"language": lang, "task": "transcribe"}} if lang else {}
            return pipe(path, chunk_length_s=30, batch_size=8, return_timestamps=True, **kwargs)

        if language:
            out = run(self._pipeline(general=language not in KHMER_NAMES), language)
        else:  # unknown language: general Whisper first; mostly Khmer → Virgo's Khmer hearing again
            out = run(self._pipeline(general=True), None)
            text = "".join(c["text"] for c in out["chunks"])
            if self.general_model and len(KHMER.findall(text)) > 0.3 * max(1, len(re.sub(r"\s", "", text))):
                out = run(self._pipeline(), "khmer")
        return [{"start": c["timestamp"][0] or 0.0, "end": c["timestamp"][1] or c["timestamp"][0] or 0.0, "text": c["text"].strip()}
                for c in out["chunks"]]

    @staticmethod
    def to_srt(segments):
        def ts(sec):
            ms = int(round(sec * 1000))
            return f"{ms // 3_600_000:02}:{ms // 60_000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"
        return "\n".join(f"{i}\n{ts(s['start'])} --> {ts(s['end'])}\n{s['text']}\n" for i, s in enumerate(segments, 1))

    # ---------- Text to speech ----------
    def tts(self, text, voice="af_heart", steps=None):
        """Text → (samples float32, sample_rate). Virgo-1.0-Angkor-Voice (VoxCPM2, vox_voice.py) speaks
        every language when it's set up; otherwise Khmer text uses the Khmer voice and English Kokoro."""
        vox = self._vox_voice()
        if vox:
            try:
                return vox(text, steps=steps)
            except Exception as err:
                print("Virgo-1.0-Angkor-Voice failed, using the other voices for now:", err)
                self._vox = False
        parts = split_languages(text)
        if len(parts) == 1:
            return self._speak_part(*parts[0], voice)
        # Khmer mixed with English: each part in its own voice (Khmer: Piseth/Sreymom, English: Andrew/Ava),
        # joined into one clip with a short pause.
        waves, rate = [], None
        for lang, part in parts:
            wave, r = self._speak_part(lang, part, voice)
            wave = np.asarray(wave, np.float32).reshape(-1)
            if rate is None:
                rate = r
            elif r != rate:
                wave = np.interp(np.arange(0, len(wave), r / rate), np.arange(len(wave)), wave).astype(np.float32)
            waves += [wave, np.zeros(int(0.06 * rate), np.float32)]
        return np.concatenate(waves[:-1]), rate

    def _speak_part(self, lang, text, voice):
        """One language's text with its fallback voice: Khmer (Piseth/Sreymom), English and other Latin-script
        languages (Andrew/Ava), else Microsoft's voice for that language (ms_voice.py)."""
        try:
            import ms_voice

            lang = ms_voice.detect_language(text, lang)  # "en" script → fr, vi, es...; "ru" → uk...
        except Exception:
            ms_voice = None
        if lang == "km":
            return self._khmer(text)
        if lang == "en":
            return self._english(text, voice)
        try:
            if ms_voice and ms_voice.available():
                return ms_voice.speak(text, ms_voice.default_voice(lang))
        except Exception as err:
            print(f"No {lang} voice, reading it with the English voice:", err)
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
        # When Virgo's voice (VoxCPM2) isn't running: Virgo's small own Khmer voice (train_voice.py) when
        # it's set up, else Microsoft's Sreymom / Piseth (ms_voice.py). Meta MMS is non-commercial
        # (CC BY-NC 4.0): only with VIRGO_ALLOW_MMS=1.
        if self._khmer_tts is None:
            self._khmer_tts = self._own_khmer_voice() or self._microsoft_voice() or self._mms_voice()
        kind, engine = self._khmer_tts
        if kind == "mms":
            return self._speak_mms(engine, text)
        try:
            return engine(text)
        except Exception as err:
            if kind == "microsoft" or not self._microsoft_voice():
                raise RuntimeError(f"Khmer voice failed: {err}") from err
            print("Virgo's Khmer voice failed, using Microsoft's for now:", err)
            self._khmer_tts = self._microsoft_voice()
            return self._khmer_tts[1](text)

    @staticmethod
    def _microsoft_voice():
        try:
            import ms_voice

            if ms_voice.available():
                print("Khmer fallback voice: Microsoft", ms_voice.voice_name(), "(Azure)" if os.environ.get("AZURE_SPEECH_KEY") else "(edge-tts)")
                return ("microsoft", ms_voice.speak)
        except Exception as err:
            print("Microsoft's Khmer voice isn't available:", err)
        return None

    def _mms_voice(self):
        if os.environ.get("VIRGO_ALLOW_MMS") != "1":
            raise RuntimeError("No Khmer voice: install edge-tts, set AZURE_SPEECH_KEY, or turn on Virgo's voice.")
        return ("mms", self._mms("facebook/mms-tts-khm"))

    @staticmethod
    def _own_khmer_voice():
        try:
            import khmer_voice

            if khmer_voice.available():
                return ("virgo", khmer_voice.KhmerVoice())
        except Exception as err:
            print("Virgo's own Khmer voice isn't available:", err)
        return None

    def _english(self, text, voice):
        # Microsoft's neural English voices (ms_voice.py: Ava, or Andrew with Bayon) sound clearest; then
        # Kokoro when it's installed. Meta's MMS English (robotic, non-commercial) only with VIRGO_ALLOW_MMS=1.
        if self._english_tts is None:
            self._english_tts = self._english_engine()
        kind, engine = self._english_tts
        if kind == "mms":
            return self._speak_mms(engine, text)
        if kind == "microsoft":
            try:
                return engine(text)
            except Exception as err:
                print("Microsoft's English voice failed, trying Kokoro:", err)
                self._english_tts = self._english_engine(skip_microsoft=True)
                return self._english(text, voice)
        parts = [audio for _, _, audio in engine(text, voice=voice)]
        return np.concatenate(parts).astype(np.float32), 24000

    def _english_engine(self, skip_microsoft=False):
        if not skip_microsoft:
            try:
                import ms_voice

                if ms_voice.available():
                    print("English fallback voice: Microsoft", ms_voice.english_voice_name())
                    return ("microsoft", ms_voice.speak_english)
            except Exception as err:
                print("Microsoft's English voice isn't available:", err)
        try:
            from kokoro import KPipeline

            return ("kokoro", KPipeline(lang_code="a"))  # American English
        except Exception:
            pass
        if os.environ.get("VIRGO_ALLOW_MMS") != "1":
            raise RuntimeError("No English voice: install edge-tts or kokoro, or set AZURE_SPEECH_KEY.")
        return ("mms", self._mms("facebook/mms-tts-eng"))

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
