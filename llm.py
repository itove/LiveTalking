import time
import os
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar
from utils.logger import logger

# Built-in LLM providers. Each exposes an OpenAI-compatible chat completions
# endpoint; the active one is selected with --llm_provider (default: dashscope).
# "openai" is any local/self-hosted server (vLLM, etc.). Aliases: vllm, custom, local.
LLM_PROVIDERS = {
    "dashscope": {
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-plus",
        "allow_empty_key": False,
    },
    "orcarouter": {
        "api_key_env": "ORCAROUTER_API_KEY",
        "base_url": "https://api.orcarouter.ai/v1",
        "default_model": "orcarouter/auto",
        "allow_empty_key": False,
    },
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "base_url": "",
        "default_model": "",
        "allow_empty_key": True,
    },
}

_PROVIDER_ALIASES = {
    "vllm": "openai",
    "custom": "openai",
    "local": "openai",
}


def _ensure_v1(url: str) -> str:
    """vLLM OpenAI clients expect a base URL that ends with /v1."""
    url = (url or "").strip().rstrip("/")
    if not url:
        return url
    if not url.endswith("/v1"):
        url += "/v1"
    return url


def _llm_provider(opt) -> str:
    """Return the configured provider name, defaulting to dashscope."""
    name = (getattr(opt, "llm_provider", "dashscope") or "dashscope").strip().lower()
    return _PROVIDER_ALIASES.get(name, name)


def _provider_cfg(opt) -> dict:
    provider = _llm_provider(opt)
    cfg = LLM_PROVIDERS.get(provider)
    if cfg is None:
        raise ValueError(
            f"Unknown --llm_provider={provider!r}. "
            "Use dashscope, orcarouter, or openai (vLLM / local Qwen)."
        )
    return cfg


def _llm_base_url(opt, cfg: dict) -> str:
    override = (
        (getattr(opt, "llm_base_url", None) or "").strip()
        or os.getenv(cfg.get("base_url_env") or "OPENAI_BASE_URL", "").strip()
    )
    if override:
        return _ensure_v1(override)
    url = cfg.get("base_url") or ""
    if not url:
        raise ValueError(
            "openai LLM provider needs --llm_base_url or OPENAI_BASE_URL "
            "(vLLM chat endpoint, usually http://<host>:<port>/v1). "
            "This is not TTS_SERVER."
        )
    return url


def _llm_api_key(cfg: dict) -> str:
    api_key = os.getenv(cfg["api_key_env"]) or ""
    if api_key:
        return api_key
    if cfg.get("allow_empty_key"):
        return "EMPTY"
    raise ValueError(
        f"Missing credentials for LLM. Set {cfg['api_key_env']}, "
        "or use --llm_provider openai with a local vLLM Qwen server."
    )


def _llm_client(opt):
    """Create the OpenAI-compatible client for the configured provider."""
    from openai import OpenAI
    cfg = _provider_cfg(opt)
    return OpenAI(
        api_key=_llm_api_key(cfg),
        base_url=_llm_base_url(opt, cfg),
    )


def _llm_model(opt) -> str:
    """Resolve the model name, falling back to the provider default."""
    cfg = _provider_cfg(opt)
    model = getattr(opt, "llm_model", "") or cfg.get("default_model") or ""
    if not model:
        raise ValueError(
            "--llm_model is required for openai/vLLM. "
            "Use the served name from GET <base>/v1/models."
        )
    return model


def llm_response(message, avatar_session: "BaseAvatar", datainfo: dict = {}):
    try:
        opt = avatar_session.opt
        start = time.perf_counter()
        client = _llm_client(opt)
        model = _llm_model(opt)
        end = time.perf_counter()
        logger.info(
            f"llm Time init: {end-start}s provider={_llm_provider(opt)} "
            f"model={model} base_url={client.base_url} {message}"
        )
        completion = client.chat.completions.create(
            model=model,
            messages=[{'role': 'system', 'content': '你是一个知识助手，尽量以简短、口语化的方式输出'},
                    {'role': 'user', 'content': message}],
            stream=True,
            # Display token usage in the last line of the streamed response.
            stream_options={"include_usage": True}
        )
        result = ""
        first = True
        for chunk in completion:
            if len(chunk.choices) > 0:
                #print(chunk.choices[0].delta.content)
                if first:
                    end = time.perf_counter()
                    logger.info(f"llm Time to first chunk: {end-start}s")
                    first = False
                msg = chunk.choices[0].delta.content
                if msg is None:
                    continue
                lastpos = 0
                #msglist = re.split('[,.!;:，。！?]',msg)
                for i, char in enumerate(msg):
                    if char in ",.!;:，。！？：；":
                        result = result + msg[lastpos:i+1]
                        lastpos = i + 1
                        if len(result) > 10:
                            logger.info(result)
                            avatar_session.put_msg_txt(result, datainfo)
                            result = ""
                result = result + msg[lastpos:]
        end = time.perf_counter()
        logger.info(f"llm Time to last chunk: {end-start}s")
        if result:
            avatar_session.put_msg_txt(result, datainfo)

    except Exception:
        logger.exception("llm exception:")
        return
