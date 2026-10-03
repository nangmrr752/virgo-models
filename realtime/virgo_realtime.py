"""Virgo 1.0 realtime speech to speech: talk to Virgo and hear it answer, over a WebSocket.

How it works (one conversation per connection):
  1. The page streams microphone audio: 16 kHz, mono, 16-bit PCM, in small binary messages.
  2. Voice activity detection (Silero VAD) notices when you start and stop talking.
  3. When you stop (0.6 s of silence), Virgo speech to text turns what you said into text.
  4. Virgo chat streams its answer; each finished sentence is spoken right away (text to speech),
     so you hear the start of the answer while the rest is still being written.
  5. Talking while Virgo speaks interrupts it.

Messages from the page: binary audio, or JSON {"type": "greet", "language": "km"|"en"} so Virgo says
hello first. Messages to the page: JSON text {"type": "heard"|"reply"|"state", ...} and binary audio
(16-bit PCM WAV bytes, one per sentence).

    python realtime/virgo_realtime.py            # ws://localhost:8765
"""
import asyncio
import io
import json
import os
import re
import sys

import numpy as np
import soundfile as sf
import torch

sys.path += [os.path.join(os.path.dirname(__file__), "..", d) for d in ("chat", "speech")]
from virgo_chat import VirgoChat  # noqa: E402
from virgo_speech import VirgoSpeech  # noqa: E402

RATE = 16000
SILENCE_S = float(os.environ.get("VIRGO_LIVE_SILENCE", "0.6"))  # a pause this long ends your turn
MIN_SPEECH_S = 0.35  # shorter sounds (a cough, a click) are ignored
FIRST_CHUNK = 40    # characters: the first words are spoken as soon as a phrase this long is ready
LONG_SENTENCE = 140  # longer sentences are spoken in parts (at a comma), so each part starts sooner
# VoxCPM2 steps in live mode: fewer is faster (the voice file setting VIRGO_VOX_STEPS stays for /v1/tts).
LIVE_STEPS = int(os.environ.get("VIRGO_LIVE_VOX_STEPS", "5"))
MAX_VOICE_TOKENS = 220
# Spoken answers: short and conversational, so Virgo starts talking sooner and sounds natural.
VOICE_SYSTEM = (
    "You are Virgo, an AI assistant made by KSN (model: {name}), talking with the user by voice. "
    "Reply in the user's language: Khmer (in Khmer script) when they speak Khmer, English when they speak English, "
    "and the same for any other language they speak. If they ask you to speak a language (\"speak English\", "
    "\"និយាយភាសាអង់គ្លេស\"), use it from then on until they ask for another. "
    "Answer in one to three short spoken sentences, like a friendly person on the phone: no lists, "
    "no markdown, no emoji. Offer more detail only if they ask. Say so when you are not sure. "
    "The user's words come from speech recognition and may hold mistakes: English words or names can be "
    "written phonetically in Khmer letters (ស្ពីងគ្លីស = \"speak English\", ហ្លូ = \"hello\", វ៉ីហ្គោ = \"Virgo\"), "
    "so read them as the English they sound like, and guess the meant words when a few letters look wrong."
)
CHECKING = {"en": "Let me check.", "km": "សូមចាំបន្តិច ខ្ញុំរកមើលសិន។"}
# What Whisper tends to "hear" in silence or noise: skipped instead of answered.
NOISE = re.compile(r"^(?:thank you for watching|thanks for watching|you|\W*|(?P<c>.)(?P=c){5,}.*)[.!。។]*$", re.I)
GREETINGS = {
    "en": "Hello! I'm Virgo AI. How can I help you?",
    "km": "សួស្តី! ខ្ញុំជា Virgo AI។ តើខ្ញុំអាចជួយអ្វីបាន?",
}
SENTENCE_END = re.compile(r"(?<=[.!?។៕\n])\s")
COMMA = re.compile(r"[,;:、،]\s*")
SPACE = re.compile(r"\s+")


def heard_nothing(text):
    """True for empty or noise-like transcripts (Whisper's usual guesses on silence)."""
    t = (text or "").strip()
    return len(t) < 2 or bool(NOISE.match(t))


def first_phrase(pending):
    """Splits off an early phrase (at a comma or space) once the first words are long enough."""
    if len(pending) < FIRST_CHUNK:
        return None, pending
    cut = None
    for pattern in (COMMA, SPACE):  # a comma is the most natural place to pause
        for m in pattern.finditer(pending, 20):
            cut = m.end()
        if cut:
            break
    if not cut or cut >= len(pending):
        return None, pending
    return pending[:cut], pending[cut:]


# ---------- Web search (through the Virgo AI website, which holds the search keys) ----------
SEARCH_URL = os.environ.get("VIRGO_SEARCH_URL", "https://virgoai.camksn.com/api/virgo-search")
CURRENT = re.compile(r"\b(today|tonight|tomorrow|yesterday|now|current(ly)?|latest|recent(ly)?|news|this (week|month|year)|right now|live|score|won|winner|price|cost|exchange rate|rate|weather|forecast|stock|election|president|prime minister|ceo|release(d)?|update|trending|open now|schedule|20[2-9]\d)\b", re.I)
KM_CURRENT = re.compile(r"(ថ្ងៃនេះ|ឥឡូវ|បច្ចុប្បន្ន|ចុងក្រោយ|ព័ត៌មាន|តម្លៃ|អាកាសធាតុ|អត្រាប្ដូរប្រាក់|អត្រា|ឆ្នាំនេះ|សប្ដាហ៍នេះ|ខែនេះ|ម្សិលមិញ|ថ្ងៃស្អែក|ពិន្ទុ|ឈ្នះ)")
UNSURE = re.compile(r"turn on \**search|i (don't|do not) know (live|the latest|today)|i can'?t (see|check) (live|today|the latest|current)|(don't|do not|can't|cannot) (have |access )?(real[- ]time|current|live|up[- ]to[- ]date)|បើក \**Search|ខ្ញុំមិនអាចមើល|ខ្ញុំមិនដឹង(លទ្ធផល|ព័ត៌មាន)ផ្ទាល់", re.I)


def wants_search(question):
    return bool(CURRENT.search(question) or KM_CURRENT.search(question))


def sounds_unsure(answer):
    return bool(UNSURE.search(answer or ""))


def web_search(query):
    """The website's search: a short text of what it found, or None (no key, offline, nothing found)."""
    import urllib.request

    key = os.environ.get("VIRGO_API_KEY")
    if not key or not SEARCH_URL:
        return None
    try:
        req = urllib.request.Request(SEARCH_URL, method="POST", data=json.dumps({"query": query[:300]}).encode(),
                                     headers={"content-type": "application/json", "authorization": f"Bearer {key}",
                                              # Cloudflare refuses Python's default "Python-urllib" (403)
                                              "user-agent": "Virgo-Server/1.0 (+https://virgoai.camksn.com)"})
        body = json.loads(urllib.request.urlopen(req, timeout=15).read())
    except Exception as err:
        print("Virgo realtime search failed:", repr(err))
        return None
    parts = [f"Summary: {body['summary']}"] if body.get("summary") else []
    for r in (body.get("results") or [])[:3]:
        parts.append(f"- {r.get('title', '')} ({r.get('url', '')}): {(r.get('text') or r.get('snippet') or '')[:1200]}")
    return "\n".join(parts) or None


def with_results(history, found):
    """The history, with the search results added to the last question (the history itself stays plain)."""
    if not found:
        return history
    last = history[-1]
    note = ("Web search results (current). Answer from them, briefly and in the user's language, and "
            f"say the source's name:\n{found}\n\nQuestion: {last['content']}")
    return history[:-1] + [{**last, "content": note}]


KHMER_TEXT = re.compile(r"[\u1780-\u17ff]")
# Asking for another language: "speak English", "in French", "និយាយភាសាអង់គ្លេស", "ជាភាសាចិន"...
SWITCH_LANGUAGE = re.compile(r"\b(speak|talk|answer|reply|switch|change)\b.{0,20}\b(english|french|chinese|thai|japanese|korean|vietnamese|spanish|german|language)\b"
                             r"|\bin (english|french|chinese|thai|japanese|korean|vietnamese|spanish|german)\b|ភាសា(អង់គ្លេស|បារាំង|ចិន|ថៃ|ជប៉ុន|កូរ៉េ|វៀតណាម)|អង់គ្លេស", re.I)


class Conversation:
    def __init__(self, chat, speech, vad):
        self.chat, self.speech, self.vad = chat, speech, vad
        self.history = []
        self.language = None  # "khmer" when the page says the user speaks Khmer; None = pick Khmer or English
        self.buffer = np.zeros(0, np.float32)
        self.speaking_task = None

    def speech_prob(self, samples):
        if self.vad is None:  # no Silero VAD: a simple loudness check
            return 1.0 if len(samples) and float(np.sqrt(np.mean(samples ** 2))) > 0.02 else 0.0
        probs = [self.vad(torch.from_numpy(samples[i:i + 512].copy()), RATE).item() for i in range(0, len(samples) - 511, 512)]
        return max(probs, default=0.0)


def split_long(sentence):
    """A long sentence → parts of about LONG_SENTENCE characters, cut at commas (else spaces)."""
    parts, rest = [], sentence
    while len(rest) > LONG_SENTENCE:
        cut = max((m.end() for m in COMMA.finditer(rest, 30, LONG_SENTENCE)), default=0) or \
            max((m.end() for m in SPACE.finditer(rest, 30, LONG_SENTENCE)), default=0)
        if not cut:
            break
        parts.append(rest[:cut])
        rest = rest[cut:]
    return parts + [rest]


async def handle(ws, chat, speech, vad, name="Angkor-1.0"):
    conv = Conversation(chat, speech, vad)
    system = VOICE_SYSTEM.format(name=name)
    talking, silent_for, utterance = False, 0.0, []

    async def send(kind, **data):
        await ws.send(json.dumps({"type": kind, **data}, ensure_ascii=False))

    async def answer(text):
        try:
            await reply_to(text)
        except asyncio.CancelledError:
            raise
        except Exception as err:  # keep the conversation going; the page falls back if it closes
            print("Virgo realtime error:", repr(err))
            await send("reply", text="Sorry, something went wrong on my side. Please say that again.")
            await send("state", state="listening")

    async def say(history, spoken):
        """Streams one answer, speaking each sentence as soon as it's written; `spoken` keeps the text."""
        loop = asyncio.get_running_loop()
        pending = ""
        stream = conv.chat.stream(history, max_new_tokens=MAX_VOICE_TOKENS, system=system)
        spoke = False
        # Voice is made in its own task while the answer keeps being written (before, writing waited
        # for each sentence's voice), and sent in order.
        queue = asyncio.Queue()

        async def voice_out():
            while (sentence := await queue.get()) is not None:
                await speak(sentence)
        voicing = asyncio.create_task(voice_out())
        await send("state", state="speaking")
        try:
            await write(loop, stream, pending, spoken, queue, spoke)
            await queue.put(None)
            await voicing
        finally:
            voicing.cancel()
        return spoken["text"]

    async def write(loop, stream, pending, spoken, queue, spoke):
        while True:
            piece = await loop.run_in_executor(None, next, stream, None)
            if piece is None:
                break
            pending += piece
            spoken["text"] += piece
            await send("reply", text=spoken["text"])
            *done, pending = SENTENCE_END.split(pending)
            if not done and not spoke:  # nothing finished yet: say the first phrase now
                phrase, pending = first_phrase(pending)
                done = [phrase] if phrase else []
            for sentence in done:
                spoke = True
                for part in split_long(sentence):
                    queue.put_nowait(part)
        if pending.strip():
            for part in split_long(pending):
                queue.put_nowait(part)

    async def reply_to(text):
        await send("state", state="thinking")
        conv.history.append({"role": "user", "content": text})
        loop = asyncio.get_running_loop()
        spoken = {"text": ""}
        try:
            # Questions about now (news, prices, weather…) are looked up first; Virgo says it's
            # checking while the search runs, so there's no silence.
            found = None
            if wants_search(text):
                lookup = loop.run_in_executor(None, web_search, text)
                await send("state", state="speaking")
                await speak(CHECKING["km" if KHMER_TEXT.search(text) else "en"])
                found = await lookup
            full = await say(with_results(conv.history, found), spoken)
            # Virgo said it can't know: look it up and answer again.
            if not found and sounds_unsure(full):
                found = await loop.run_in_executor(None, web_search, text)
                if found:
                    spoken["text"] = ""
                    await send("state", state="thinking")
                    await say(with_results(conv.history, found), spoken)
        finally:
            # Keep what Virgo said, even when interrupted or failed, so turns stay user/assistant.
            if spoken["text"].strip():
                conv.history.append({"role": "assistant", "content": spoken["text"].strip()})
        await send("state", state="listening")

    async def speak(sentence):
        if not sentence.strip():
            return
        wave, rate = await asyncio.get_running_loop().run_in_executor(None, lambda: conv.speech.tts(sentence.strip(), steps=LIVE_STEPS))
        out = io.BytesIO()
        sf.write(out, wave, rate, format="WAV", subtype="PCM_16")
        await ws.send(out.getvalue())

    async def greet(language):
        text = GREETINGS.get(language, GREETINGS["en"])
        await send("state", state="speaking")
        await send("reply", text=text)
        await speak(text)
        await send("state", state="listening")

    await send("state", state="listening")
    greeted = False
    async for message in ws:
        if isinstance(message, str):
            # {"type": "greet", "language": "km" | "en"}: the page asks Virgo to say hello first.
            try:
                event = json.loads(message)
            except ValueError:
                continue
            # {"type": "language", "language": "km"}: the user chose Khmer, so hear only Khmer (no guessing).
            if isinstance(event, dict) and event.get("type") == "language":
                conv.language = {"km": "khmer", "en": "english"}.get(event.get("language"))
            if isinstance(event, dict) and event.get("type") == "greet" and not greeted:
                greeted = True
                conv.speaking_task = asyncio.create_task(greet(event.get("language")))
            continue
        samples = np.frombuffer(message, np.int16).astype(np.float32) / 32768
        voice = conv.speech_prob(samples) > 0.5
        if voice:
            if not talking:
                # You started talking: stop Virgo's voice on the page (even if the answer is already
                # fully sent and just playing), and stop writing the rest of the answer.
                await send("interrupted")
                if conv.speaking_task and not conv.speaking_task.done():
                    conv.speaking_task.cancel()
                    await send("state", state="listening")
            talking, silent_for = True, 0.0
        elif talking:
            silent_for += len(samples) / RATE
        if talking:
            utterance.append(samples)
        if talking and silent_for >= SILENCE_S:
            audio = np.concatenate(utterance)
            talking, silent_for, utterance = False, 0.0, []
            if len(audio) / RATE - SILENCE_S < MIN_SPEECH_S:
                continue
            try:
                text = await asyncio.get_running_loop().run_in_executor(None, conv.speech.stt, {"raw": audio, "sampling_rate": RATE}, conv.language)
            except Exception as err:  # one bad turn never ends the conversation
                print("Virgo realtime speech-to-text error:", repr(err))
                await send("reply", text="Sorry, I didn't catch that. Please say it again.")
                await send("state", state="listening")
                continue
            # "Speak English" / "និយាយភាសាអង់គ្លេស" (or another language): stop listening for Khmer only.
            if text and conv.language and SWITCH_LANGUAGE.search(text):
                conv.language = None
            if text and not heard_nothing(text):
                await send("heard", text=text)
                conv.speaking_task = asyncio.create_task(answer(text))


def load_vad():
    """Silero VAD (downloaded once); None means a simple loudness check is used instead."""
    try:
        vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
        return vad
    except Exception as err:
        print("Silero VAD unavailable, using a loudness check:", err)
        return None


async def main():
    import websockets

    adapter = "chat/out/virgo-1.0-chat-lora"
    chat = VirgoChat(adapter=adapter if os.path.isdir(adapter) else None)
    speech = VirgoSpeech()
    vad = load_vad()
    port = int(os.environ.get("PORT", 8765))
    async with websockets.serve(lambda ws: handle(ws, chat, speech, vad), "0.0.0.0", port, max_size=2**20):
        print(f"Virgo 1.0 realtime voice on ws://localhost:{port}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
