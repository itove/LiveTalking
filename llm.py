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


_client_cache = {}


def _llm_client(opt):
    """Reuse one OpenAI client per provider URL so every chat is not a 1s+ handshake."""
    from openai import OpenAI
    cfg = _provider_cfg(opt)
    base_url = _llm_base_url(opt, cfg)
    key = (_llm_provider(opt), base_url)
    client = _client_cache.get(key)
    if client is None:
        client = OpenAI(
            api_key=_llm_api_key(cfg),
            base_url=base_url,
        )
        _client_cache[key] = client
    return client


def _as_bool(value, default=False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _thinking_enabled(opt) -> bool:
    return _as_bool(getattr(opt, "llm_enable_thinking", False), default=False)


def _chat_extra_body(opt):
    """Kwargs that are not OpenAI-standard (vLLM / Qwen chat template).

    Qwen3.8 defaults to reasoning_effort=xhigh whenever thinking is left on.
    Spoken avatars need the hard switch off so the first audible token is not
    delayed by a think block.
    """
    if _llm_provider(opt) != "openai":
        return None
    if _thinking_enabled(opt):
        return None
    return {"chat_template_kwargs": {"enable_thinking": False}}


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


# Flush TTS at sentence end. Commas only flush once the clause is already
# long enough that Edge's ~7s round-trip can hide behind playback.
_SENTENCE_END = frozenset("。！？.!?")
_CLAUSE_PAUSE = frozenset("，,、；;：:")
_MIN_SENTENCE_CHARS = 10
_MIN_CLAUSE_CHARS = 40


def take_spoken_segments(buffer: str, incoming: str) -> tuple[list[str], str]:
    """Split streamed LLM text into speakable chunks.

    Tiny comma-separated clauses each paid a full TTS round-trip and left
    the avatar silent while the next clip was fetched.
    """
    flushed: list[str] = []
    lastpos = 0
    result = buffer
    for i, char in enumerate(incoming):
        if char in _SENTENCE_END or char in _CLAUSE_PAUSE:
            result = result + incoming[lastpos:i + 1]
            lastpos = i + 1
            min_chars = _MIN_SENTENCE_CHARS if char in _SENTENCE_END else _MIN_CLAUSE_CHARS
            if len(result) > min_chars:
                flushed.append(result)
                result = ""
    result = result + incoming[lastpos:]
    return flushed, result


def llm_response(message, avatar_session: "BaseAvatar", datainfo: dict = {}):
    try:
        opt = avatar_session.opt
        start = time.perf_counter()
        client = _llm_client(opt)
        model = _llm_model(opt)
        end = time.perf_counter()
        extra_body = _chat_extra_body(opt)
        logger.info(
            f"llm Time init: {end-start}s provider={_llm_provider(opt)} "
            f"model={model} base_url={client.base_url} "
            f"thinking={_thinking_enabled(opt)} {message}"
        )
        create_kwargs = {
            "model": model,
            "messages": [
                {'role': 'system', 'content': '你是一个知识助手，尽量以简短、口语化的方式输出，不要使用markdown。第一句话先用十个字以内点题，后面再展开。'},
                {'role': 'user', 'content': message},
            ],
            "stream": True,
            # Display token usage in the last line of the streamed response.
            "stream_options": {"include_usage": True},
        }
        if extra_body:
            create_kwargs["extra_body"] = extra_body
        completion = client.chat.completions.create(**create_kwargs)
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
                pieces, result = take_spoken_segments(result, msg)
                for piece in pieces:
                    logger.info(piece)
                    avatar_session.put_msg_txt(piece, datainfo)
        end = time.perf_counter()
        logger.info(f"llm Time to last chunk: {end-start}s")
        if result:
            avatar_session.put_msg_txt(result, datainfo)

    except Exception:
        logger.exception("llm exception:")
        return
