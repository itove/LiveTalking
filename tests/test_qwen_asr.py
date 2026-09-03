import io
import json
import types
import unittest
import wave
from unittest.mock import patch

import numpy as np

from server.asr_server import (
    funasr_result_payload,
    is_qwen_asr_configured,
    _transcribe_buffer,
)
from server.qwen_asr import pcm16_to_wav_bytes, transcribe_pcm16


class FakeTranscriptionsResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.content = json.dumps(payload).encode("utf-8")
        self.text = json.dumps(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class QwenAsrClientTests(unittest.TestCase):
    def test_pcm16_to_wav_is_mono_16k(self):
        pcm = np.zeros(1600, dtype=np.int16).tobytes()
        wav = pcm16_to_wav_bytes(pcm, sample_rate=16000)
        self.assertTrue(wav.startswith(b"RIFF"))
        with wave.open(io.BytesIO(wav), "rb") as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getsampwidth(), 2)
            self.assertEqual(wf.getframerate(), 16000)
            self.assertEqual(wf.readframes(wf.getnframes()), pcm)

    def test_transcribe_pcm16_posts_wav_and_model(self):
        pcm = np.arange(320, dtype=np.int16).tobytes()
        captured = {}

        def fake_post(url, files=None, data=None, headers=None, timeout=None):
            captured["url"] = url
            captured["files"] = files
            captured["data"] = data
            captured["headers"] = headers
            captured["timeout"] = timeout
            return FakeTranscriptionsResponse({"text": "  hello world  "})

        with patch("server.qwen_asr.requests.post", side_effect=fake_post):
            text = transcribe_pcm16(
                pcm,
                "http://asr.example:8010/",
                model="qwen3-asr",
                language="zh",
                api_key="EMPTY",
            )

        self.assertEqual(text, "hello world")
        self.assertEqual(captured["url"], "http://asr.example:8010/v1/audio/transcriptions")
        self.assertEqual(captured["data"]["model"], "qwen3-asr")
        self.assertEqual(captured["data"]["language"], "zh")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer EMPTY")
        filename, wav_bytes, mime = captured["files"]["file"]
        self.assertEqual(filename, "speech.wav")
        self.assertEqual(mime, "audio/wav")
        self.assertTrue(wav_bytes.startswith(b"RIFF"))

    def test_transcribe_empty_server_raises(self):
        with self.assertRaises(ValueError):
            transcribe_pcm16(b"\x00\x00", "")


class FunasrAdapterTests(unittest.TestCase):
    def test_funasr_2pass_offline_payload(self):
        payload = funasr_result_payload("你好", "2pass")
        self.assertEqual(payload["text"], "你好")
        self.assertEqual(payload["mode"], "2pass-offline")
        self.assertTrue(payload["is_final"])
        self.assertIsNone(payload["timestamp"])
        json.dumps(payload)

    def test_is_qwen_asr_configured(self):
        self.assertFalse(is_qwen_asr_configured(None))
        self.assertFalse(is_qwen_asr_configured(types.SimpleNamespace(ASR_SERVER="")))
        self.assertTrue(
            is_qwen_asr_configured(types.SimpleNamespace(ASR_SERVER="http://asr.example:8010"))
        )

    def test_transcribe_buffer_qwen_to_funasr_json(self):
        pcm = np.zeros(1600, dtype=np.int16).tobytes()
        opt = types.SimpleNamespace(
            ASR_SERVER="http://asr.example:8010",
            asr_model="qwen3-asr",
            asr_language="",
        )

        def fake_post(url, files=None, data=None, headers=None, timeout=None):
            self.assertEqual(url, "http://asr.example:8010/v1/audio/transcriptions")
            self.assertEqual(data.get("model"), "qwen3-asr")
            wav_bytes = files["file"][1]
            self.assertTrue(wav_bytes.startswith(b"RIFF"))
            return FakeTranscriptionsResponse({"text": "你好世界"})

        with patch("server.qwen_asr.requests.post", side_effect=fake_post):
            text, inference_ms, audio_dur = _transcribe_buffer(pcm, 16000, opt, False)

        self.assertEqual(text, "你好世界")
        self.assertGreater(audio_dur, 0)
        self.assertGreaterEqual(inference_ms, 0)
        payload = funasr_result_payload(text, "2pass")
        self.assertEqual(payload["mode"], "2pass-offline")
        self.assertEqual(payload["text"], "你好世界")


if __name__ == "__main__":
    unittest.main()
