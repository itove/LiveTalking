import io
import unittest

import av
import numpy as np

from tts.edge import iter_pcm_from_mp3, _mono_f32


def _sine_mp3(seconds=0.4, sr=24000) -> bytes:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    wave = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buf = io.BytesIO()
    out = av.open(buf, mode="w", format="mp3")
    stream = out.add_stream("mp3", rate=sr)
    stream.layout = "mono"
    frame = av.AudioFrame.from_ndarray(
        wave.reshape(1, -1), format="flt", layout="mono"
    )
    frame.sample_rate = sr
    for packet in stream.encode(frame):
        out.mux(packet)
    for packet in stream.encode(None):
        out.mux(packet)
    out.close()
    return buf.getvalue()


class EdgeMp3StreamTests(unittest.TestCase):
    def test_iter_pcm_from_mp3_yields_16k(self):
        mp3 = _sine_mp3()
        chunks = list(iter_pcm_from_mp3(io.BytesIO(mp3), 16000))
        self.assertTrue(chunks)
        pcm = np.concatenate(chunks)
        self.assertEqual(pcm.dtype, np.float32)
        self.assertGreater(pcm.shape[0], 16000 * 0.3)
        self.assertLess(pcm.shape[0], 16000 * 0.6)

    def test_mono_f32_flattens_planar(self):
        samples = np.zeros((1, 64), dtype=np.float32)
        frame = av.AudioFrame.from_ndarray(samples, format="flt", layout="mono")
        frame.sample_rate = 16000
        out = _mono_f32(frame)
        self.assertEqual(out.ndim, 1)
        self.assertEqual(out.shape[0], 64)


if __name__ == "__main__":
    unittest.main()
