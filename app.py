"""Your own text-to-speech server, powered by Kokoro (open-source, Apache 2.0).

POST /tts   {"text": "...", "voice": "af_heart", "speed": 1.0}  ->  audio/wav
GET  /voices, GET /health
Set the TTS_API_KEY environment variable to require "Authorization: Bearer <key>".
"""
import hmac
import io
import os
import re
import threading

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from kokoro import KPipeline
from pydantic import BaseModel, Field

API_KEY = os.environ.get("TTS_API_KEY", "")
SAMPLE_RATE = 24000
VOICES = ["af_heart", "af_bella", "af_nicole", "af_sky", "am_adam", "am_michael", "bf_emma", "bm_george"]

app = FastAPI(title="My TTS")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_pipelines = {}
_lock = threading.Lock()  # the model handles one request at a time


def get_pipeline(lang: str) -> KPipeline:
    if lang not in _pipelines:
        _pipelines[lang] = KPipeline(lang_code=lang)
    return _pipelines[lang]


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    voice: str = "af_heart"
    speed: float = Field(1.0, ge=0.6, le=1.4)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/voices")
def voices():
    return {"voices": VOICES}


@app.post("/tts")
def tts(req: TTSRequest, authorization: str | None = Header(default=None)):
    if API_KEY and not hmac.compare_digest(authorization or "", f"Bearer {API_KEY}"):
        raise HTTPException(status_code=401, detail="Bad API key")
    if not re.fullmatch(r"[a-z]{2}_[a-z]+", req.voice):
        raise HTTPException(status_code=400, detail="Bad voice name")
    lang = req.voice[0] if req.voice[0] in ("a", "b") else "a"

    parts = []
    gap = np.zeros(int(0.12 * SAMPLE_RATE), dtype=np.float32)
    with _lock:
        for _, _, audio in get_pipeline(lang)(req.text, voice=req.voice, speed=req.speed, split_pattern=r"\n+"):
            if audio is None:
                continue
            arr = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
            parts.extend([arr.astype(np.float32), gap])
    if not parts:
        raise HTTPException(status_code=400, detail="No speech produced")

    buf = io.BytesIO()
    sf.write(buf, np.concatenate(parts[:-1]), SAMPLE_RATE, format="WAV")
    return Response(content=buf.getvalue(), media_type="audio/wav")
