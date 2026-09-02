import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import llm


def _opt(**kwargs):
    defaults = {
        "llm_provider": "dashscope",
        "llm_model": "",
        "llm_base_url": "",
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class LlmClientTests(unittest.TestCase):
    def test_ensure_v1_appends_once(self):
        self.assertEqual(llm._ensure_v1("http://host:8000"), "http://host:8000/v1")
        self.assertEqual(llm._ensure_v1("http://host:8000/v1"), "http://host:8000/v1")
        self.assertEqual(llm._ensure_v1("http://host:8000/v1/"), "http://host:8000/v1")

    def test_vllm_alias_is_openai(self):
        self.assertEqual(llm._llm_provider(_opt(llm_provider="vllm")), "openai")
        self.assertEqual(llm._llm_provider(_opt(llm_provider="custom")), "openai")

    def test_unknown_provider_raises(self):
        with self.assertRaises(ValueError) as ctx:
            llm._provider_cfg(_opt(llm_provider="not-a-vendor"))
        self.assertIn("Unknown --llm_provider", str(ctx.exception))

    def test_dashscope_missing_key_raises(self):
        env = {k: v for k, v in os.environ.items() if k != "DASHSCOPE_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError) as ctx:
                llm._llm_api_key(llm.LLM_PROVIDERS["dashscope"])
        self.assertIn("DASHSCOPE_API_KEY", str(ctx.exception))

    def test_openai_empty_key_becomes_empty_token(self):
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(
                llm._llm_api_key(llm.LLM_PROVIDERS["openai"]), "EMPTY"
            )

    def test_openai_requires_base_url(self):
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("OPENAI_BASE_URL", "OPENAI_API_KEY")
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError) as ctx:
                llm._llm_base_url(_opt(llm_provider="openai"), llm.LLM_PROVIDERS["openai"])
        self.assertIn("llm_base_url", str(ctx.exception))

    def test_openai_model_required(self):
        with self.assertRaises(ValueError) as ctx:
            llm._llm_model(_opt(llm_provider="openai", llm_model=""))
        self.assertIn("--llm_model", str(ctx.exception))

    @patch("openai.OpenAI")
    def test_openai_client_uses_cli_url_and_dummy_key(self, mock_openai):
        mock_openai.return_value = MagicMock()
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("OPENAI_BASE_URL", "OPENAI_API_KEY")
        }
        with patch.dict(os.environ, env, clear=True):
            llm._llm_client(_opt(
                llm_provider="openai",
                llm_base_url="http://127.0.0.1:8000",
                llm_model="Qwen3-27B",
            ))
        mock_openai.assert_called_once_with(
            api_key="EMPTY",
            base_url="http://127.0.0.1:8000/v1",
        )

    def test_dashscope_default_model(self):
        self.assertEqual(llm._llm_model(_opt()), "qwen-plus")

    def test_openai_disables_qwen_thinking_by_default(self):
        extra = llm._chat_extra_body(_opt(llm_provider="openai"))
        self.assertEqual(
            extra, {"chat_template_kwargs": {"enable_thinking": False}}
        )

    def test_openai_thinking_opt_in_skips_extra_body(self):
        self.assertIsNone(llm._chat_extra_body(_opt(
            llm_provider="openai",
            llm_enable_thinking=True,
        )))

    def test_dashscope_does_not_send_qwen_template_kwargs(self):
        self.assertIsNone(llm._chat_extra_body(_opt()))


if __name__ == "__main__":
    unittest.main()
