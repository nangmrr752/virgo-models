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
VIRGO_KHMER_FALLBACK=off turns this off.
"""
import asyncio
import io
import os
import urllib.request
from xml.sax.saxutils import escape

VOICES = {"sreymom": "km-KH-SreymomNeural", "piseth": "km-KH-PisethNeural"}


def voice_name():
    wanted = os.environ.get("VIRGO_KHMER_VOICE") or ("piseth" if os.environ.get("VIRGO_MAIN", "").lower() == "bayon" else "sreymom")
    return VOICES.get(wanted.lower(), wanted)  # a short name, or a full Microsoft voice name


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
    ssml = (f"<speak version='1.0' xml:lang='km-KH'><voice name='{voice}'>{escape(text)}</voice></speak>").encode("utf-8")
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
        async for chunk in edge_tts.Communicate(text, voice).stream():
            if chunk["type"] == "audio":
                audio += chunk["data"]
        return bytes(audio)

    data = asyncio.run(run())  # tts runs in a worker thread, so there's no event loop here
    if not data:
        raise RuntimeError("edge-tts returned no audio")
    return _decode(data)


def speak(text, voice=None):
    """Khmer text → (samples float32, sample_rate)."""
    voice = voice or voice_name()
    return _azure(text, voice) if os.environ.get("AZURE_SPEECH_KEY") else _edge(text, voice)
