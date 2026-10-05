"""Virgo 1.0 API: every ability behind one HTTP server, ready for whichever machine runs it.

    uvicorn serve.app:app --host 0.0.0.0 --port 8000

  POST /v1/chat        {"messages": [...], "model": "virgo-1.0-bayon"}    → {"reply": "..."}  (model: optional;
                       "stream": true → NDJSON lines {"t": "..."} as it writes)
  POST /v1/tts         {"text": "សួស្តី", "voice": "af_heart"}              → audio/wav
  POST /v1/stt         audio file (form field "file"), ?language=khmer     → {"text": "..."}
  POST /v1/transcribe  audio file (form field "file"), ?srt=1              → {"segments": [...]} or .srt
  POST /v1/images      {"prompt": "...", "steps": 2, "size": 512}          → image/png
  POST /v1/videos      {"prompt": "...", "seconds": 5}                     → video/mp4 (GPU only)
  WS   /v1/realtime    realtime speech to speech (see realtime/virgo_realtime.py)
  GET  /v1/models      what this server offers

One server can run several Virgo chat models (VIRGO_CHAT_MODELS="virgo-1.0-angkor=<folder>,virgo-1.0-bayon=<folder>");
GET /v1/models lists them under "chat_models", and /v1/chat picks one by "model" (the first by default).
Models load the first time they're used, so the server starts fast and only uses memory for the
abilities people actually call. Set VIRGO_API_KEY to require "Authorization: Bearer <key>".
"""
import io
import json
import threading
import os
import sys
import tempfile

import soundfile as sf
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile, WebSocket
from fastapi.responses import PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path += [os.path.join(ROOT, d) for d in ("chat", "speech", "image", "video", "realtime")]

app = FastAPI(title="Virgo", version="1.0.0")
_models = {}
_locks = {}


def chat_models():
    """{model id: folder}, in order (the first is the default, used by real-time voice too)."""
    spec = os.environ.get("VIRGO_CHAT_MODELS", "")
    found = {}
    for part in filter(None, (p.strip() for p in spec.split(","))):
        name, _, folder = part.partition("=")
        folder = folder if os.path.isabs(folder) else os.path.join(ROOT, folder)
        if name and os.path.isdir(folder):
            found[name.strip()] = folder
    return found or {"virgo-1.0-angkor": os.path.join(ROOT, "chat/out/virgo-1.0-chat-lora")}


def load_chat(model=None):
    """A Virgo chat model by id. With several GPUs each model gets its own (the first model the first
    GPU, the next the last GPU, …); on one GPU they share it."""
    import torch

    models = chat_models()
    model = model or next(iter(models))
    if model not in models and model.startswith("virgo-") and os.environ.get("VIRGO_MAIN", "").lower() == "bayon":
        model = next(iter(models))  # Bayon alone (VIRGO_MAIN=bayon): it answers for Angkor too
    if model not in models:
        raise HTTPException(404, f"No chat model {model!r} here. This server has: {', '.join(models)}.")
    key = f"chat:{model}"
    # VIRGO_LLAMA_URL: the main model (the first) answers through llama.cpp's server instead of being loaded
    # here, e.g. Angkor-2.0 on Gemma 4 26B A4B (docker-compose.yml, "llama" service). Its name: VIRGO_LLAMA_NAME.
    llama = os.environ.get("VIRGO_LLAMA_URL", "").strip()
    if llama and model == next(iter(models)) and key not in _models:
        from llama_chat import LlamaChat

        _models[key] = LlamaChat(llama, name=os.environ.get("VIRGO_LLAMA_NAME", "Angkor-2.0").strip() or "Angkor-2.0")
        return _models[key]
    if key not in _models:
        from virgo_chat import VirgoChat

        gpus = torch.cuda.device_count()
        index = list(models).index(model)
        gpu = 0 if gpus <= 1 or index == 0 else gpus - 1 - (index - 1) % (gpus - 1)
        folder = models[model]
        name = "Bayon-1.0" if "bayon" in model else "Angkor-1.0"
        _models[key] = VirgoChat(adapter=folder if os.path.isdir(folder) else None, gpu=gpu, name=name)
        print(f"Chat model {model} ready on GPU {gpu}")
    return _models[key]


@app.on_event("startup")
def preload():
    """VIRGO_PRELOAD=chat,speech loads those models at start, so the first request isn't slow."""
    names = [n.strip() for n in os.environ.get("VIRGO_PRELOAD", "").split(",") if n.strip()]
    # Speech (and Virgo's voice) first: the voice needs the most free memory while it loads. With a
    # big chat model (VIRGO_MAIN=bayon, 27B) the chat goes first instead, so it always starts; if the
    # voice then doesn't fit, the fallback voices speak and the server keeps running.
    chat_first = os.environ.get("VIRGO_MAIN", "").lower() == "bayon"
    for name in sorted(names, key=lambda n: n != ("chat" if chat_first else "speech")):
        load(name)
        if name == "chat":
            for model in list(chat_models())[1:]:
                load_chat(model)
        if name == "speech":
            # Start Virgo's own voice (VoxCPM2) now: the log says right away whether it works, and the
            # first answer isn't slow.
            try:  # both ears now (every-language Whisper + Virgo's Khmer hearing), not on the first question
                _models["speech"]._pipeline()
                _models["speech"]._pipeline(general=True)
            except Exception as err:
                print("Hearing didn't preload:", err)
            voice = _models["speech"]._vox_voice()
            name = (os.environ.get("VIRGO_VOX_REPO") or "Virgo's voice").split("/")[-1]
            print("🗣️ Voice:", f"{name} (VoxCPM2)" if voice else f"fallback voices (Microsoft / Kokoro): {name} isn't set up or didn't start")


def load(name):
    if name not in _models:
        if name == "chat":
            _models[name] = load_chat()
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
    model: str | None = None
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False  # true: NDJSON lines {"t": "piece"} as the answer is written


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
        info = json.load(f)
    ids = list(chat_models())
    # Each model's name (the website shows it): the llama.cpp model's VIRGO_LLAMA_NAME (Angkor-2.0), else Angkor-1.0 / Bayon-1.0.
    llama_name = (os.environ.get("VIRGO_LLAMA_NAME", "").strip() or "Angkor-2.0") if os.environ.get("VIRGO_LLAMA_URL", "").strip() else ""
    names = {m: (llama_name if i == 0 and llama_name else "Bayon-1.0" if "bayon" in m else "Angkor-1.0") for i, m in enumerate(ids)}
    return {**info, "chat_models": ids, "chat_names": names} if isinstance(info, dict) else info


@app.post("/v1/chat", dependencies=[Depends(auth)])
def chat(body: ChatBody):
    if not body.messages:
        raise HTTPException(400, "No messages.")
    bot = load_chat(body.model)
    lock = _locks.setdefault(id(bot), threading.Lock())  # one answer at a time per model: no GPU contention
    max_tokens = min(body.max_tokens, 2048)
    if body.stream:
        def pieces():
            with lock:
                for piece in bot.stream(body.messages, max_tokens, body.temperature):
                    if piece:
                        yield json.dumps({"t": piece}, ensure_ascii=False) + "\n"
        return StreamingResponse(pieces(), media_type="application/x-ndjson")
    with lock:
        reply = bot.reply(body.messages, max_tokens, body.temperature)
    return {"model": body.model or next(iter(chat_models())), "reply": reply}


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
    from virgo_realtime import handle, load_vad

    if "vad" not in _models:
        _models["vad"] = load_vad()

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

    # ?model=virgo-1.0-bayon: that model talks (the server's first model when not given or not here).
    model = ws.query_params.get("model")
    model = model if model in chat_models() else next(iter(chat_models()))
    # The live voice has its own name: Bakong-{VIRGO_LIVE_VERSION} (2.0), answering with Angkor or Bayon.
    chat_name = getattr(load_chat(model), "name", None) or ("Bayon-1.0" if "bayon" in model else "Angkor-1.0")
    name = f"Bakong-{os.environ.get('VIRGO_LIVE_VERSION', '2.0').strip()}, KSN's real-time voice, answering with {chat_name}"
    await handle(Adapter(), load_chat(model), load("speech"), _models["vad"], name)
