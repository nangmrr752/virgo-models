"""Virgo 1.0 API: every ability behind one HTTP server, ready for whichever machine runs it.

    uvicorn serve.app:app --host 0.0.0.0 --port 8000

  POST /v1/chat        {"messages": [{"role": "user", "content": "Hi"}]}   → {"reply": "..."}
  POST /v1/tts         {"text": "សួស្តី", "voice": "af_heart"}              → audio/wav
  POST /v1/stt         audio file (form field "file"), ?language=khmer     → {"text": "..."}
  POST /v1/transcribe  audio file (form field "file"), ?srt=1              → {"segments": [...]} or .srt
  POST /v1/images      {"prompt": "...", "steps": 2, "size": 512}          → image/png
  POST /v1/videos      {"prompt": "...", "seconds": 5}                     → video/mp4 (GPU only)
  WS   /v1/realtime    realtime speech to speech (see realtime/virgo_realtime.py)
  GET  /v1/models      what this server offers

Models load the first time they're used, so the server starts fast and only uses memory for the
abilities people actually call. Set VIRGO_API_KEY to require "Authorization: Bearer <key>".
"""
import io
import os
import sys
import tempfile

import soundfile as sf
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile, WebSocket
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path += [os.path.join(ROOT, d) for d in ("chat", "speech", "image", "video", "realtime")]

app = FastAPI(title="Virgo 1.0", version="1.0.0")
_models = {}


def load(name):
    if name not in _models:
        if name == "chat":
            from virgo_chat import VirgoChat

            adapter = os.path.join(ROOT, "chat/out/virgo-1.0-chat-lora")
            _models[name] = VirgoChat(adapter=adapter if os.path.isdir(adapter) else None)
        elif name == "speech":
            from virgo_speech import VirgoSpeech

            _models[name] = VirgoSpeech()
        elif name == "image":
            from virgo_image import VirgoImage

            _models[name] = VirgoImage()
        elif name == "video":
            from virgo_video import VirgoVideo

            _models[name] = VirgoVideo()
    return _models[name]


def auth(authorization: str = Header(default="")):
    key = os.environ.get("VIRGO_API_KEY")
    if key and authorization != f"Bearer {key}":
        raise HTTPException(401, "Missing or wrong API key.")


class ChatBody(BaseModel):
    messages: list[dict]
    max_tokens: int = 512
    temperature: float = 0.7


class TtsBody(BaseModel):
    text: str
    voice: str = "af_heart"


class ImageBody(BaseModel):
    prompt: str
    steps: int = 2
    size: int = 512
    seed: int | None = None


class VideoBody(BaseModel):
    prompt: str
    seconds: float = 5
    seed: int | None = None


async def saved(upload: UploadFile):
    suffix = os.path.splitext(upload.filename or "")[1] or ".wav"
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    f.write(await upload.read())
    f.close()
    return f.name


@app.get("/v1/models")
def models():
    import json

    with open(os.path.join(ROOT, "virgo.json"), encoding="utf-8") as f:
        return json.load(f)


@app.post("/v1/chat", dependencies=[Depends(auth)])
def chat(body: ChatBody):
    if not body.messages:
        raise HTTPException(400, "No messages.")
    return {"model": "virgo-1.0", "reply": load("chat").reply(body.messages, min(body.max_tokens, 2048), body.temperature)}


@app.post("/v1/tts", dependencies=[Depends(auth)])
def tts(body: TtsBody):
    if not body.text.strip() or len(body.text) > 3000:
        raise HTTPException(400, "Text must be 1 to 3000 characters.")
    wave, rate = load("speech").tts(body.text, body.voice)
    out = io.BytesIO()
    sf.write(out, wave, rate, format="WAV", subtype="PCM_16")
    return Response(out.getvalue(), media_type="audio/wav")


@app.post("/v1/stt", dependencies=[Depends(auth)])
async def stt(file: UploadFile = File(...), language: str | None = None):
    path = await saved(file)
    try:
        return {"text": load("speech").stt(path, language)}
    finally:
        os.remove(path)


@app.post("/v1/transcribe", dependencies=[Depends(auth)])
async def transcribe(file: UploadFile = File(...), language: str | None = None, srt: bool = False):
    path = await saved(file)
    try:
        speech = load("speech")
        segments = speech.transcribe(path, language)
    finally:
        os.remove(path)
    if srt:
        return PlainTextResponse(speech.to_srt(segments), media_type="application/x-subrip")
    return {"text": " ".join(s["text"] for s in segments), "segments": segments}


@app.post("/v1/images", dependencies=[Depends(auth)])
def images(body: ImageBody):
    image = load("image").generate(body.prompt, max(1, min(body.steps, 8)), min(body.size, 768), body.seed)
    out = io.BytesIO()
    image.save(out, format="PNG")
    return Response(out.getvalue(), media_type="image/png")


@app.post("/v1/videos", dependencies=[Depends(auth)])
def videos(body: VideoBody):
    from diffusers.utils import export_to_video

    frames, fps = load("video").generate(body.prompt, max(1, min(body.seconds, 8)), seed=body.seed)
    path = tempfile.mktemp(suffix=".mp4")
    export_to_video(frames, path, fps=fps)
    try:
        with open(path, "rb") as f:
            return Response(f.read(), media_type="video/mp4")
    finally:
        os.remove(path)


@app.websocket("/v1/realtime")
async def realtime(ws: WebSocket):
    key = os.environ.get("VIRGO_API_KEY")
    if key and ws.query_params.get("key") != key:
        await ws.close(code=1008)
        return
    await ws.accept()
    import torch
    from virgo_realtime import handle

    if "vad" not in _models:
        _models["vad"], _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)

    class Adapter:  # gives FastAPI's WebSocket the small interface the realtime handler uses
        async def send(self, data):
            await (ws.send_bytes(data) if isinstance(data, bytes) else ws.send_text(data))

        def __aiter__(self):
            return self

        async def __anext__(self):
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                raise StopAsyncIteration
            return message.get("bytes") or message.get("text") or ""

    await handle(Adapter(), load("chat"), load("speech"), _models["vad"])
