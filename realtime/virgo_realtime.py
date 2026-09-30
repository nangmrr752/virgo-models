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
SILENCE_S = 0.6
GREETINGS = {
    "en": "Hello! I'm Virgo AI. How can I help you?",
    "km": "សួស្តី! ខ្ញុំជា Virgo AI។ តើខ្ញុំអាចជួយអ្វីបាន?",
}
SENTENCE_END = re.compile(r"(?<=[.!?។៕\n])\s")


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


async def handle(ws, chat, speech, vad):
    conv = Conversation(chat, speech, vad)
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

    async def reply_to(text):
        await send("state", state="thinking")
        conv.history.append({"role": "user", "content": text})
        loop = asyncio.get_running_loop()
        pending, full = "", ""
        try:
            stream = conv.chat.stream(conv.history)
            await send("state", state="speaking")
            while True:
                piece = await loop.run_in_executor(None, next, stream, None)
                if piece is None:
                    break
                pending += piece
                full += piece
                await send("reply", text=full)
                *done, pending = SENTENCE_END.split(pending)
                for sentence in done:
                    await speak(sentence)
            if pending.strip():
                await speak(pending)
        finally:
            # Keep what Virgo said, even when interrupted or failed, so turns stay user/assistant.
            if full.strip():
                conv.history.append({"role": "assistant", "content": full.strip()})
        await send("state", state="listening")

    async def speak(sentence):
        if not sentence.strip():
            return
        wave, rate = await asyncio.get_running_loop().run_in_executor(None, conv.speech.tts, sentence.strip())
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
            try:
                text = await asyncio.get_running_loop().run_in_executor(None, conv.speech.stt, {"raw": audio, "sampling_rate": RATE}, conv.language)
            except Exception as err:  # one bad turn never ends the conversation
                print("Virgo realtime speech-to-text error:", repr(err))
                await send("reply", text="Sorry, I didn't catch that. Please say it again.")
                await send("state", state="listening")
                continue
            if text:
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
