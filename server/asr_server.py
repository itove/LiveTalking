###############################################################################
#  ASR WebSocket Server — FunASR client protocol
#
#  Resolves: https://github.com/lipku/LiveTalking/issues/604
#
#  WebSocket /api/asr speaks the same protocol as FunASR
#  (wss://www.funasr.com:10096/). The browser (web/asr/main.js) is unchanged.
#
#  Transcription backend:
#    --ASR_SERVER  →  Qwen3-ASR POST /v1/audio/transcriptions (preferred)
#    otherwise     →  local SenseVoice if funasr is installed
#
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  Licensed under the Apache License, Version 2.0
###############################################################################

import json
import time
import io
import asyncio
import threading
import numpy as np
from aiohttp import web

from utils.logger import logger


# ─── Lazy Model Loader ────────────────────────────────────────────────────

_sensevoice_model = None
_sensevoice_load_lock = threading.Lock()
_sensevoice_inference_lock = threading.Lock()


def _load_sensevoice():
    """
    Load the SenseVoice model on first call (lazy singleton).
    Concurrent first requests must share the same model initialization.
    """
    global _sensevoice_model
    if _sensevoice_model is not None:
        return _sensevoice_model

    with _sensevoice_load_lock:
        if _sensevoice_model is not None:
            return _sensevoice_model

        import torch
        from funasr import AutoModel

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(
            f"[ASR] Loading SenseVoiceSmall on device='{device}' "
            f"(first run will download ~500MB from ModelScope)..."
        )

        t0 = time.perf_counter()
        _sensevoice_model = AutoModel(
            model="iic/SenseVoiceSmall",
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device=device,
            trust_remote_code=True,
        )
        elapsed = time.perf_counter() - t0
        logger.info(
            f"[ASR] ✅ SenseVoiceSmall ready — loaded in {elapsed:.1f}s on {device}"
        )
    return _sensevoice_model


def _run_inference(audio_float32: np.ndarray, sample_rate: int, use_itn: bool):
    """
    Run SenseVoice inference on a float32 audio array.

    This is a **blocking** call — always invoke from ``run_in_executor``.

    Returns
    -------
    tuple[str, float, float]
        (transcribed_text, inference_ms, audio_duration_s)
    """
    import soundfile as sf
    from funasr.utils.postprocess_utils import rich_transcription_postprocess

    model = _load_sensevoice()

    # Write to in-memory WAV so funasr can read the sample rate from the header
    wav_buf = io.BytesIO()
    sf.write(wav_buf, audio_float32, sample_rate, format="WAV")
    wav_buf.seek(0)

    t0 = time.perf_counter()
    with _sensevoice_inference_lock:
        res = model.generate(
            input=wav_buf,
            cache={},
            language="auto",
            use_itn=use_itn,
            batch_size_s=60,
        )
    inference_ms = (time.perf_counter() - t0) * 1000

    text = ""
    if res and len(res) > 0 and res[0].get("text"):
        text = rich_transcription_postprocess(res[0]["text"])

    audio_duration_s = len(audio_float32) / sample_rate

    logger.info(
        f"[ASR] ✅ SenseVoice inference complete\n"
        f"       ├─ Latency     : {inference_ms:>8.0f} ms\n"
        f"       ├─ Audio length: {audio_duration_s:>8.1f} s\n"
        f"       ├─ RTF         : {inference_ms / 1000 / max(audio_duration_s, 0.001):>8.3f}\n"
        f"       └─ Text        : \"{text[:100]}{'…' if len(text) > 100 else ''}\""
    )

    return text, inference_ms, audio_duration_s


SAMPLE_RATE = 16000  # The browser client records at 16 kHz mono PCM16


def is_qwen_asr_configured(opt) -> bool:
    if opt is None:
        return False
    return bool((getattr(opt, "ASR_SERVER", "") or "").strip())


def funasr_result_payload(text: str, client_mode: str = "2pass") -> dict:
    """JSON the FunASR web client expects after a full utterance."""
    if client_mode == "2pass":
        response_mode = "2pass-offline"
    else:
        response_mode = client_mode or "offline"
    return {
        "text": text,
        "mode": response_mode,
        "is_final": True,
        "timestamp": None,
    }


def _run_qwen_inference(pcm: bytes, sample_rate: int, opt):
    from server.qwen_asr import transcribe_pcm16

    t0 = time.perf_counter()
    text = transcribe_pcm16(
        pcm,
        getattr(opt, "ASR_SERVER", ""),
        model=getattr(opt, "asr_model", "") or "",
        language=getattr(opt, "asr_language", "") or "",
    )
    inference_ms = (time.perf_counter() - t0) * 1000
    audio_duration_s = len(pcm) / (sample_rate * 2)
    logger.info(
        f"[ASR] Qwen inference complete latency={inference_ms:.0f}ms "
        f"audio={audio_duration_s:.1f}s text={text[:100]!r}"
    )
    return text, inference_ms, audio_duration_s


def _transcribe_buffer(pcm: bytes, sample_rate: int, opt, use_itn: bool):
    """Blocking transcription. Qwen remote ASR if configured, else local SenseVoice."""
    if is_qwen_asr_configured(opt):
        return _run_qwen_inference(pcm, sample_rate, opt)
    audio_float32 = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return _run_inference(audio_float32, sample_rate, use_itn)


# ─── WebSocket Handler ─────────────────────────────────────────────────────


async def asr_websocket_handler(request):
    """
    WebSocket handler implementing the FunASR client protocol.

    Protocol flow
    -------------
    1. Client opens connection
    2. Client sends JSON config::

           {"chunk_size":[5,10,5], "wav_name":"h5",
            "is_speaking":true, "mode":"2pass", "itn":false, ...}

    3. Client streams binary PCM16 audio chunks (960 bytes = 60 ms @ 16 kHz)
    4. Client sends stop signal::

           {"is_speaking":false, ...}

    5. Server responds with transcription::

           {"text":"hello world", "mode":"2pass-offline",
            "is_final":true, "timestamp":null}
    """
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    client_ip = request.remote
    logger.info(f"[ASR] 🔌 WebSocket connected from {client_ip}")

    audio_buffer = bytearray()
    config: dict = {}
    session_start = time.perf_counter()
    chunks_received = 0

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    logger.warning("[ASR] Received invalid JSON, ignoring")
                    continue

                if data.get("is_speaking") is True:
                    # ── Session start ──────────────────────────────────
                    config = data
                    audio_buffer = bytearray()
                    chunks_received = 0
                    session_start = time.perf_counter()
                    logger.info(
                        f"[ASR] 🎙️  Recording started | "
                        f"mode={config.get('mode', 'offline')} | "
                        f"itn={config.get('itn', False)} | "
                        f"hotwords={bool(config.get('hotwords'))}"
                    )

                elif data.get("is_speaking") is False:
                    # ── End of speech → run inference ──────────────────
                    buf_bytes = len(audio_buffer)
                    audio_seconds = buf_bytes / (SAMPLE_RATE * 2)  # 2 bytes per int16
                    session_elapsed = time.perf_counter() - session_start

                    logger.info(
                        f"[ASR] 🛑 Recording stopped | "
                        f"{chunks_received} chunks | "
                        f"{buf_bytes:,} bytes | "
                        f"{audio_seconds:.1f}s audio | "
                        f"session wall time {session_elapsed:.1f}s"
                    )

                    if buf_bytes < 640:  # < 20 ms of audio — skip
                        logger.warning("[ASR] Audio too short (< 20ms), returning empty")
                        await ws.send_str(json.dumps({
                            "text": "",
                            "mode": config.get("mode", "offline"),
                            "is_final": True,
                            "timestamp": None,
                        }))
                        continue

                    # Ensure even number of bytes for int16 conversion
                    if buf_bytes % 2 != 0:
                        logger.warning(f"[ASR] Odd number of bytes received ({buf_bytes}), dropping incomplete sample")
                        audio_buffer = audio_buffer[:-1]
                        buf_bytes -= 1

                    pcm = bytes(audio_buffer)
                    use_itn = config.get("itn", False)
                    opt = request.app.get("opt") if request.app else None

                    loop = asyncio.get_event_loop()
                    try:
                        text, inference_ms, audio_dur = await loop.run_in_executor(
                            None,
                            _transcribe_buffer,
                            pcm,
                            SAMPLE_RATE,
                            opt,
                            use_itn,
                        )
                    except Exception as e:
                        logger.exception(f"[ASR] ❌ Inference failed: {e}")
                        text = ""

                    payload = funasr_result_payload(text, config.get("mode", "offline"))
                    await ws.send_str(json.dumps(payload))
                    logger.info(f"[ASR] 📤 Result sent to client (mode={payload['mode']})")

            elif msg.type == web.WSMsgType.BINARY:
                audio_buffer.extend(msg.data)
                chunks_received += 1

            elif msg.type in (web.WSMsgType.ERROR, web.WSMsgType.CLOSE):
                break

    except asyncio.CancelledError:
        logger.info("[ASR] WebSocket handler cancelled")
    except Exception as e:
        logger.exception(f"[ASR] ❌ WebSocket handler error: {e}")

    logger.info(f"[ASR] 🔌 WebSocket disconnected ({client_ip})")
    return ws


# ─── Availability Check ───────────────────────────────────────────────────

def is_funasr_available() -> bool:
    """Return True if the ``funasr`` package is importable."""
    try:
        import funasr  # noqa: F401
        return True
    except ImportError:
        return False
