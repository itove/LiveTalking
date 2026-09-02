import time
import asyncio
import numpy as np
import resampy
import soundfile as sf
import edge_tts
from io import BytesIO

from utils.logger import logger
from .base_tts import BaseTTS, State
from registry import register

@register("tts", "edgetts")
class EdgeTTS(BaseTTS):
    # Overlap the next Microsoft round-trip with current playback.
    synth_workers = 2

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

    async def _stream_into(self, voicename: str, text: str, buf: BytesIO):
        try:
            communicate = edge_tts.Communicate(text, voicename)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio" and self.state == State.RUNNING:
                    buf.write(chunk["data"])
        except Exception:
            logger.exception("edgetts")
