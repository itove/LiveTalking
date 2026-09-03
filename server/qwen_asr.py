"""Qwen3-ASR client: OpenAI-compatible POST /v1/audio/transcriptions."""

import io
import os
import time
import wave

import requests

from utils.logger import logger

SAMPLE_RATE = 16000


def pcm16_to_wav_bytes(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wrap raw little-endian PCM16 mono as a WAV file."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def transcribe_pcm16(
    pcm: bytes,
    server_url: str,
    model: str = "",
    language: str = "",
    api_key: str = "",
    timeout: float = 120,
) -> str:
    """POST a WAV clip to vLLM Qwen3-ASR. Returns transcribed text (may be empty)."""
    base = (server_url or "").strip().rstrip("/")
    if not base:
        raise ValueError("ASR_SERVER is empty")
    url = base + "/v1/audio/transcriptions"
    wav = pcm16_to_wav_bytes(pcm)
    files = {"file": ("speech.wav", wav, "audio/wav")}
    data = {}
    if model:
        data["model"] = model
    if language:
        data["language"] = language
    key = api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
    headers = {"Authorization": f"Bearer {key}"}
    t0 = time.perf_counter()
    res = requests.post(url, files=files, data=data, headers=headers, timeout=timeout)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    if res.status_code >= 400:
        logger.error("Qwen ASR HTTP %s: %s", res.status_code, res.text[:500])
        res.raise_for_status()
    body = res.json() if res.content else {}
    text = (body.get("text") or "").strip()
    logger.info(
        f"[ASR] Qwen transcriptions {elapsed_ms:.0f}ms "
        f"bytes={len(pcm)} text={text[:80]!r}"
    )
    return text
