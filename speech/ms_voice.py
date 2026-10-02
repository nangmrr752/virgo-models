"""Microsoft's Khmer voices, Sreymom (km-KH-SreymomNeural, a woman) and Piseth (km-KH-PisethNeural, a man):
Virgo's fallback Khmer voice when Virgo's own voice (VoxCPM2) isn't running. Replaces Meta MMS, which is
non-commercial (CC BY-NC 4.0).

Two ways to reach them, the same voices either way:
  - Azure AI Speech, the official paid service: set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION (e.g.
    southeastasia). Licensed for commercial use. Used whenever the key is set.
  - edge-tts (the default without a key): Microsoft Edge's free "Read aloud" service, used unofficially.
    It isn't licensed for commercial use and can stop working without notice: move to an Azure key for
    a public product.
VIRGO_KHMER_VOICE picks the voice (default Piseth with Bayon, VIRGO_MAIN=bayon, else Sreymom).
VIRGO_KHMER_RATE (e.g. -10% slower, +5% faster) and VIRGO_KHMER_PITCH (e.g. -4Hz, +4Hz) change how it sounds.
VIRGO_KHMER_FALLBACK=off turns this off.
"""
import asyncio
import io
import os
import urllib.request
from xml.sax.saxutils import escape

VOICES = {"sreymom": "km-KH-SreymomNeural", "piseth": "km-KH-PisethNeural",
          # English: Microsoft's clearest, most natural neural voices
          "ava": "en-US-AvaMultilingualNeural", "andrew": "en-US-AndrewMultilingualNeural",
          "emma": "en-US-EmmaMultilingualNeural", "brian": "en-US-BrianMultilingualNeural"}


def english_voice_name():
    """VIRGO_ENGLISH_FALLBACK_VOICE: ava (a woman, the default) / andrew (a man, the default with Bayon) /
    emma / brian, or any full Microsoft voice name."""
    bayon = os.environ.get("VIRGO_MAIN", "").lower() == "bayon"
    wanted = os.environ.get("VIRGO_ENGLISH_FALLBACK_VOICE") or ("andrew" if bayon else "ava")
    return VOICES.get(wanted.lower(), wanted)


def voice_name():
    wanted = os.environ.get("VIRGO_KHMER_VOICE") or ("piseth" if os.environ.get("VIRGO_MAIN", "").lower() == "bayon" else "sreymom")
    return VOICES.get(wanted.lower(), wanted)  # a short name, or a full Microsoft voice name


def _signed(value, default):
    value = (value or "").strip() or default
    return value if value[0] in "+-" else "+" + value  # edge-tts needs the sign: "10%" → "+10%"


def _rate():
    return _signed(os.environ.get("VIRGO_KHMER_RATE"), "+0%")


def _pitch():
    return _signed(os.environ.get("VIRGO_KHMER_PITCH"), "+0Hz")


def available():
    if os.environ.get("VIRGO_KHMER_FALLBACK", "on").lower() == "off":
        return False
    if os.environ.get("AZURE_SPEECH_KEY"):
        return True
    try:
        import edge_tts  # noqa: F401

        return True
    except ImportError:
        return False


def _decode(data):
    import soundfile as sf

    try:
        wave, rate = sf.read(io.BytesIO(data), dtype="float32")
    except Exception:  # an older libsndfile without MP3
        import librosa

        wave, rate = librosa.load(io.BytesIO(data), sr=None)
    return (wave.mean(axis=1) if wave.ndim > 1 else wave), rate


def _azure(text, voice):
    region = os.environ.get("AZURE_SPEECH_REGION", "southeastasia")
    lang = "-".join(voice.split("-")[:2])
    ssml = (f"<speak version='1.0' xml:lang='{lang}'><voice name='{voice}'><prosody rate='{_rate()}' pitch='{_pitch()}'>"
            f"{escape(text)}</prosody></voice></speak>").encode("utf-8")
    req = urllib.request.Request(
        f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1", data=ssml, method="POST",
        headers={"Ocp-Apim-Subscription-Key": os.environ["AZURE_SPEECH_KEY"], "Content-Type": "application/ssml+xml",
                 "X-Microsoft-OutputFormat": "riff-24khz-16bit-mono-pcm", "User-Agent": "virgo"})
    with urllib.request.urlopen(req, timeout=30) as res:
        return _decode(res.read())


def _edge(text, voice):
    import edge_tts

    async def run():
        audio = bytearray()
        async for chunk in edge_tts.Communicate(text, voice, rate=_rate(), pitch=_pitch()).stream():
            if chunk["type"] == "audio":
                audio += chunk["data"]
        return bytes(audio)

    data = asyncio.run(run())  # tts runs in a worker thread, so there's no event loop here
    if not data:
        raise RuntimeError("edge-tts returned no audio")
    return _decode(data)


def speak(text, voice=None):
    """Khmer text (or any text, with `voice` naming another language's voice) → (samples float32, sample_rate)."""
    voice = voice or voice_name()
    return _azure(text, voice) if os.environ.get("AZURE_SPEECH_KEY") else _edge(text, voice)


def speak_english(text):
    """English text → (samples, rate) with a clear Microsoft neural voice (Ava, or Andrew with Bayon)."""
    return speak(text, english_voice_name())


# One woman's and one man's voice for each other language the fallback can tell apart by its script.
OTHER_VOICES = {
    "th": ("th-TH-PremwadeeNeural", "th-TH-NiwatNeural"), "lo": ("lo-LA-KeomanyNeural", "lo-LA-ChanthavongNeural"),
    "my": ("my-MM-NilarNeural", "my-MM-ThihaNeural"), "zh": ("zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural"),
    "ja": ("ja-JP-NanamiNeural", "ja-JP-KeitaNeural"), "ko": ("ko-KR-SunHiNeural", "ko-KR-InJoonNeural"),
    "hi": ("hi-IN-SwaraNeural", "hi-IN-MadhurNeural"), "ar": ("ar-SA-ZariyahNeural", "ar-SA-HamedNeural"),
    "ru": ("ru-RU-SvetlanaNeural", "ru-RU-DmitryNeural"),
}


def voice_for(lang):
    """Microsoft's voice for a language code: a man's with Bayon (VIRGO_MAIN=bayon), else a woman's."""
    if lang == "km":
        return voice_name()
    if lang == "en":
        return english_voice_name()
    woman, man = OTHER_VOICES.get(lang, (None, None))
    if not woman:
        return english_voice_name()  # the multilingual English voice reads many languages
    return man if os.environ.get("VIRGO_MAIN", "").lower() == "bayon" else woman
