import os
import time
import asyncio
import threading
from io import BytesIO

import av
import numpy as np
import resampy
import soundfile as sf
import edge_tts

from utils.logger import logger
from .base_tts import BaseTTS, State
from registry import register


def _mono_f32(frame: av.AudioFrame) -> np.ndarray:
    arr = frame.to_ndarray()
    if arr.ndim == 2:
        arr = arr[0] if arr.shape[0] <= 8 else arr[:, 0]
    return np.ascontiguousarray(arr, dtype=np.float32)


def iter_pcm_from_mp3(src, sample_rate: int):
    """Yield 16 kHz mono float32 as MP3 bytes arrive on a file-like `src`."""
    inp = av.open(src, format="mp3", mode="r")
    resampler = av.AudioResampler(format="flt", layout="mono", rate=sample_rate)
    try:
        for frame in inp.decode(audio=0):
            resampled = resampler.resample(frame)
            frames = resampled if isinstance(resampled, (list, tuple)) else [resampled]
            for rf in frames:
                if rf is None:
                    continue
                samples = _mono_f32(rf)
                if samples.size:
                    yield samples
        flushed = resampler.resample(None)
        frames = flushed if isinstance(flushed, (list, tuple)) else [flushed]
        for rf in frames:
            if rf is None:
                continue
            samples = _mono_f32(rf)
            if samples.size:
                yield samples
    finally:
        inp.close()


@register("tts", "edgetts")
class EdgeTTS(BaseTTS):
    # Overlap the next Microsoft round-trip with current playback.
    synth_workers = 2

    def txt_to_audio(self, msg: tuple[str, dict]):
        """Stream the current sentence so first audio is not delayed ~7s."""
        text, textevent = msg
        voice = self.opt.REF_FILE or "zh-CN-YunxiaNeural"
        voicename = textevent.get("tts", {}).get("ref_file", voice)
        r_fd, w_fd = os.pipe()
        t0 = time.perf_counter()
        pump = threading.Thread(
            target=self._pump_mp3,
            args=(voicename, text, w_fd),
            daemon=True,
        )
        pump.start()
        first = True
        leftover = np.zeros(0, dtype=np.float32)
        try:
            with os.fdopen(r_fd, "rb", buffering=0) as src:
                for samples in iter_pcm_from_mp3(src, self.sample_rate):
                    if self.state != State.RUNNING:
                        break
                    leftover = np.concatenate([leftover, samples])
                    while leftover.shape[0] >= self.chunk and self.state == State.RUNNING:
                        eventpoint = {}
                        if first:
                            logger.info(
                                f"edge tts first pcm: {time.perf_counter() - t0:.3f}s"
                            )
                            eventpoint = {"status": "start", "text": text}
                            first = False
                        eventpoint.update(**textevent)
                        self.parent.put_audio_frame(leftover[: self.chunk], eventpoint)
                        leftover = leftover[self.chunk :]
        except Exception:
            logger.exception("edgetts stream decode")
        finally:
            pump.join(timeout=5)

        eventpoint = {"status": "end", "text": text}
        eventpoint.update(**textevent)
        self.parent.put_audio_frame(
            np.zeros(self.chunk, dtype=np.float32), eventpoint
        )

    def _pump_mp3(self, voicename: str, text: str, w_fd: int):
        try:
            with os.fdopen(w_fd, "wb", buffering=0) as sink:
                asyncio.run(self._stream_into(voicename, text, sink))
        except BrokenPipeError:
            pass
        except Exception:
            logger.exception("edgetts mp3 pump")
            try:
                os.close(w_fd)
            except OSError:
                pass

    def synthesize(self, msg: tuple[str, dict]):
        text, textevent = msg
        voice = self.opt.REF_FILE or "zh-CN-YunxiaNeural"
        voicename = textevent.get("tts", {}).get("ref_file", voice)
        t = time.time()
        buf = BytesIO()
        asyncio.run(self._stream_into(voicename, text, buf))
        logger.info(f"-------edge tts time:{time.time()-t:.4f}s")
        if buf.getbuffer().nbytes <= 0:
            logger.error("edgetts err!!!!!")
            return None
        buf.seek(0)
        return self._decode_pcm(buf)

    def _decode_pcm(self, byte_stream):
        stream, sample_rate = sf.read(byte_stream)  # [T*sample_rate,] float64
        logger.info(f"[INFO]tts audio stream {sample_rate}: {stream.shape}")
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f"[WARN] audio has {stream.shape[1]} channels, only use the first.")
            stream = stream[:, 0]

        if sample_rate != self.sample_rate and stream.shape[0] > 0:
            logger.info(f"[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.")
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream

    async def _stream_into(self, voicename: str, text: str, buf):
        try:
            communicate = edge_tts.Communicate(text, voicename)
            async for chunk in communicate.stream():
                if self.state != State.RUNNING:
                    break
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
        except BrokenPipeError:
            return
        except Exception:
            logger.exception("edgetts")
